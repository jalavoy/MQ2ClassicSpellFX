// MQ2ClassicSpellFX - puts the classic (spells.eff) hand effects back on the caster's hands.
//
// This plugin hooks the spell-cast effect spawner and the old-style emitter creator:
//  - code 0: one emitter per hand, attached to the hand's attach node. EQGraphics then moves the
//    emitter with the bone and orients it with the bone every frame (old-style emitter
//    definitions always use orient-to-attach mode), so the spray follows the forearm.
//    A few hand emitters have a spawn radius (a disc around the emitter) sized for the body;
//    it is scaled down so the disc stays inside the hand.
//
// Only the cast spawner is hooked. The persistent object-effect spawner keeps its emitter handle,
// and attached emitters are deleted by EQGraphics when the actor's nodes are torn down, so
// attaching there would leave that handle dangling.
//
// RoF2 (emulator) builds: the RoF2 client's spawner still creates one emitter per hand and asks each
// handle to attach itself, but EQGraphicsDX9.dll ignores the request. There the plugin answers that
// request instead of creating the emitters (see the RoF2 section below).
//
// Spell effect records, actor fields and the EQGraphics emitter types come from eqlib. The two hooked
// functions are found with a byte pattern at load. If either isn't found exactly once the plugin stays inactive.

#include <mq/Plugin.h>

#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>
#include <utility>

// eqlib's IS_EMU_CLIENT covers every emulator client. ROF2EMU is its define for a RoF-level (RoF2) build only.
#if defined(ROF2EMU)
#define IS_ROF2_CLIENT 1
#else
#define IS_ROF2_CLIENT 0
#endif

PreSetup("MQ2ClassicSpellFX");
PLUGIN_VERSION(0.8);

namespace {

#if IS_ROF2_CLIENT
// RoF2's eqlib names for the same spells.eff structures.
using SpellEffect = OldSpellEffect;
using EQ_SpellExtra = EQSpellExtra;
#endif

//============================================================================
// Re-finding these values after a patch
//============================================================================
//
// Everything below was derived from the 2026-10-21 live client (eqgame.exe and EQGraphics.dll of that
// date). Addresses quoted as "was 0x14..." / "was 0x18..." are from that build at the preferred image
// base; they WILL move, so use them only to orient yourself in an old disassembly. The anchors that
// survive patches are strings, distinctive constants (0xAE8, 0x3A0) and the call relationships
// described here. The RoF2 values are from the RoF2 client (eqgame.exe and EQGraphicsDX9.dll dated
// 2019-05-27), quoted as "RoF2: 0x..." at the preferred image base. That client doesn't change.
//
// Tools that worked:
//  - Ghidra (string search, xrefs, disassembly). The decompiler may fail on this binary; disassembly is enough.
//  - objdump -d -M intel -j .text eqgame.exe > text.asm, then grep. Fast for "find every imul by 0x3A0".
//
// When a signature fails at load, the plugin prints its name. Find the function again with the steps in
// that signature's comment, then rebuild the byte array and mask from the new bytes (wildcard RIP-relative
// displacements: 0x00 in the array, '?' in the mask) and check it is unique in eqgame.exe.

constexpr int EffSectionCount = static_cast<int>(std::extent_v<decltype(SpellEffect::stages)>);
constexpr int EffEmittersPerSection = static_cast<int>(std::extent_v<decltype(StageType::Radius)>);

// Not an offset: a tuning value. Scale for the spawn radius of hand emitters. Only two hand (code 0)
// emitters in spells.eff have a radius: record 2 / cast emitter 1 (Shielding's fire, GENC10_SPRITE, 5)
// and record 41 / cast emitter 0 (bard songs, GENC20_SPRITE, 3). In classic the disc fit inside the hand.
constexpr float HandSpawnRadiusScale = 0.05f;

//----------------------------------------------------------------------------
// Location code -> preset index ("mode")
//----------------------------------------------------------------------------
// The spawner reads each emitter's location code from `[record + 0x94 + section*0x3A0 + i*4]`
// (0x94 = header 8 + 0x8C), checks `cmp eax, 5 / ja`, and jumps through a 6-entry table. Each case
// ends by storing a constant into the first stack argument of CreateOldEmitter (`mov dword [rsp+0x20], N`)
// and calling it. On the analysed build the mapping was:
//   code 0 -> 0  (no position computed: the position buffer stays 0,0,0; the bug this plugin fixes)
//   code 1 -> 5  (uses the impact-point argument if non-null, else 0,0,0)
//   code 2 -> 1  (impact point, else extra+0x10, else actor position)
//   code 3 -> 3, code 4 -> 4, code 5 -> 6 (actor position snapped to the ground)
// The constant is also the index into a table of 7 hard-coded particle presets (a .bss array of 7
// pointers filled by one function that allocates 0x88-byte blocks seven times), and it appears as the
// last %d in the "%s_%d_%d_%d_%d" emitter name.
// RoF2: same table and mapping, but the code 0 case calls CreateOldEmitter twice (once per hand) and passes
// each handle an attach point (see the RoF2 section).
constexpr int ModeHands = 0;         // location code 0

#if !IS_ROF2_CLIENT
//----------------------------------------------------------------------------
// Actor (eqgame.exe)
//----------------------------------------------------------------------------
// The spawner's actors are PlayerClient. Their position and attach points (mActorClient.pParticlePoints)
// come from eqlib.
//
// AttachSlotHands: the new-system spawner (was 0x1402882f0, called by SpawnCastEffect when the spell has a
// spellsnew record) has an origin switch with a row of `mov edx, N / mov rcx, rbx / call <getter>`, where
// the getter reads pParticlePoints[N]. New-system origin 04 passes slot 3 and origin 05 passes slot 4. The
// model setup function (it references "%sR_POINT_DAG" and "%sL_POINT_DAG") binds slot 3 to R_POINT_DAG
// and slot 4 to L_POINT_DAG (look for `mov edx, 3` / `mov edx, 4` before a virtual call that takes the
// formatted name). The P99 community labels origin 04 as the left hand, so R/L may be mirrored; it
// doesn't matter because both hands are used.
constexpr int AttachSlotHands[] = { 3, 4 }; // R_POINT_DAG, L_POINT_DAG

//----------------------------------------------------------------------------
// EQGraphics.dll
//----------------------------------------------------------------------------
// CreateOldEmitter ends in CSceneGraph::CreateParticleCloudInstance (`call [rax+0xF8]`, was 0x18009d200),
// which goes through CParticleSystem::CreateSpellEmitter. That path always creates the emitter without an
// attach point, so the plugin sets CEmitter::pAttachPoint itself afterwards.
#else
//----------------------------------------------------------------------------
// RoF2: EQGraphicsDX9.dll
//----------------------------------------------------------------------------
// The code 0 case in RoF2's spawner (RoF2: 0x521dc9) creates one emitter per hand. After each one it looks up
// a bone on the actor (ActorClient::pActor, CActorInterface vtable +0x1C0 by name, +0x1BC by number):
// "<model>L_POINT_DAG" (then "R_" for the second), else "<model>SPELL_POINT_DAG", else a numbered bone,
// and calls CParticleCloudInterface::AttachToBone(bone) on the handle. The handle is a CParticleCloud (0xC bytes,
// RTTI ".?AVCParticleCloud@@", vtable RoF2: 0x10139b88, created by CSceneGraph::CreateParticleCloudInstance at
// scene graph vtable +0x78), and its AttachToBone (vtable slot 1, RoF2: 0x1003b040) is a stub:
// `or eax, -1 / ret 4`. So the emitters stay at 0,0,0. The plugin points that vtable entry at AttachCloudToBone
// while it's installed.
//  - The per-frame update (RoF2: 0x10072548, it writes CEmitter::Position) follows CEmitter::pAttachBone (its
//    translation at bone +0xD0), else pActor, else pAttachPoint.
//  - CParticleSystem's remove-by-actor (RoF2: 0x10070a60) deletes every emitter whose actor is the one being
//    removed, or whose bone belongs to it (bone vtable +0x14).
constexpr int CloudAttachSlot = 1; // CParticleCloudInterface::AttachToBone
const uint8_t CloudAttachStub[] = { 0x83, 0xC8, 0xFF, 0xC2, 0x04, 0x00 }; // or eax, -1 / ret 4
#endif

struct Signature
{
	const char* name;
	const uint8_t* pattern;
	const char* mask; // eqlib FindPattern mask: 'x' = match, '?' = wildcard
};

#if !IS_ROF2_CLIENT
// SpawnCastEffect (eqgame.exe, was 0x140287be0): spawns a spell's effect for one section (cast, middle,
// landing).
//   int SpawnCastEffect(int section, EQ_Spell*, EQ_SpellExtra*, PlayerClient* caster, PlayerClient* target,
//                       const CVector3* impactPoint, void* extra, int)
// How to find it:
//  - EQ_SpellExtra is {OldSpellEff, NewSpellEff}. The spawner checks the new pointer first and calls the
//    new-system spawner if it is set; otherwise it runs the old path.
//  - The old path walks 3 emitters with `imul reg, section, 0x3A0`, compares the tag with "Source" and
//    "Target", and switches on the location code (see the mode notes above).
//  - There is a near-identical twin (was 0x140286d60) used for persistent effects on objects. It is
//    called from an actor/object update loop that does `imul reg, idx, 0xAE8` against the spells.eff
//    table, and from one hard-coded splash effect. Do NOT hook the twin (see the file header).
//    SpawnCastEffect is the one with about 5 callers, which pass the section in ecx (0 or 2) and
//    0 or a vec3 pointer as the 6th argument.
//  - Prologue on the analysed build: `movsxd r15, ecx / mov r12, r9 / mov rbx, r8 / mov rdi, rdx /
//    test r9, r9 / je` (caster null check first).
const uint8_t SpawnCastPattern[] = {
	0x40, 0x55, 0x53, 0x57, 0x41, 0x54, 0x41, 0x57, 0x48, 0x8D, 0x6C, 0x24,
	0xE9, 0x48, 0x81, 0xEC, 0xE0, 0x00, 0x00, 0x00, 0x4C, 0x63, 0xF9, 0x4D,
	0x8B, 0xE1, 0x49, 0x8B, 0xD8, 0x48, 0x8B, 0xFA, 0x4D, 0x85, 0xC9, 0x0F,
	0x84
};
const Signature SpawnCastSig = { "SpawnCastEffect", SpawnCastPattern,
	"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" };

// CreateOldEmitter (eqgame.exe, was 0x1402911d0): creates one old-style emitter.
//   CParticleCloudInterface* CreateOldEmitter(SpellEffect*, int id, int section, int emitterIndex,
//                                             int mode, const CVector3& position, int duration)
// How to find it:
//  - It is the only code that references the format string "%s_%d_%d_%d_%d" (the emitter name).
//  - It indexes the 7-entry preset table with the mode: `movsxd rsi, [rbp+0x150]` then
//    `mov r9, [table + rsi*8]`. The trailing "48 63 B5 50 01 00 00" in the pattern is that mode load,
//    so if the stack layout changes the pattern fails rather than reading the wrong argument.
//  - It does `imul rcx, rdi, 0x3A0`, reads the per-emitter overrides at [r8 + r14*4 + 0x22C..0x298],
//    and ends with a virtual call `call [rax+0xF8]` on the scene graph (see the EQGraphics note).
//  - Both spawners call it at the end of every location-code case.
const uint8_t CreateOldEmitterPattern[] = {
	0x40, 0x55, 0x56, 0x57, 0x41, 0x54, 0x41, 0x56, 0x41, 0x57, 0x48, 0x8D,
	0xAC, 0x24, 0x08, 0xFF, 0xFF, 0xFF, 0x48, 0x81, 0xEC, 0xF8, 0x01, 0x00,
	0x00, 0x48, 0x8B, 0x05, 0x00, 0x00, 0x00, 0x00, 0x48, 0x33, 0xC4, 0x48,
	0x89, 0x85, 0xE0, 0x00, 0x00, 0x00, 0x48, 0x63, 0xB5, 0x50, 0x01, 0x00,
	0x00
};
const Signature CreateOldEmitterSig = { "CreateOldEmitter", CreateOldEmitterPattern,
	"xxxxxxxxxxxxxxxxxxxxxxxxxxxx????xxxxxxxxxxxxxxxxx" };
#else
// RoF2 SpawnCastEffect (RoF2: 0x522500): same arguments as live, cdecl. It checks the caster, the spell and
// EQSpellExtra {OldSpellEff, NewSpellEff}, takes the new-system spawner (RoF2: 0x51aa20) if NewSpellEff is
// set, and otherwise calls the old spawner (RoF2: 0x521ca0). The old spawner is the one that references
// "Source" and "Target" and does `imul eax, eax, 0x3A0`; it is also called directly for persistent object
// effects (`imul eax, eax, 0xAE8` before the call), which don't go through SpawnCastEffect. Do NOT hook it.
// The pattern is the prologue with its null checks; jump displacements are wildcarded.
const uint8_t SpawnCastPattern[] = {
	0x53, 0x55, 0x8B, 0x6C, 0x24, 0x18, 0x56, 0x85, 0xED, 0x0F, 0x84, 0x00,
	0x00, 0x00, 0x00, 0x83, 0x7C, 0x24, 0x14, 0x00, 0x0F, 0x84, 0x00, 0x00,
	0x00, 0x00, 0x8B, 0x5C, 0x24, 0x18, 0x85, 0xDB, 0x0F, 0x84, 0x00, 0x00,
	0x00, 0x00, 0x8B, 0x0B, 0x85, 0xC9, 0x75, 0x09, 0x39, 0x4B, 0x04, 0x0F,
	0x84
};
const Signature SpawnCastSig = { "SpawnCastEffect", SpawnCastPattern,
	"xxxxxxxxxxx????xxxxxxx????xxxxxxxx????xxxxxxxxxxx" };

// RoF2 CreateOldEmitter (RoF2: 0x51a810): same arguments as live, cdecl, position passed by pointer.
// It is the only code that references "%s_%d_%d_%d_%d", and it ends with `call [eax+0x78]` on the scene graph.
// The pattern ends with the mode load `mov edi, [esp+0x1A8]` and the preset table read
// `mov eax, [edi*4 + table]` (table address wildcarded).
const uint8_t CreateOldEmitterPattern[] = {
	0x81, 0xEC, 0x88, 0x01, 0x00, 0x00, 0x8B, 0x94, 0x24, 0x8C, 0x01, 0x00,
	0x00, 0x53, 0x8B, 0x9C, 0x24, 0x98, 0x01, 0x00, 0x00, 0x8B, 0xCB, 0x69,
	0xC9, 0xA0, 0x03, 0x00, 0x00, 0x56, 0x57, 0x8B, 0xBC, 0x24, 0xA8, 0x01,
	0x00, 0x00, 0x8B, 0x04, 0xBD, 0x00, 0x00, 0x00, 0x00
};
const Signature CreateOldEmitterSig = { "CreateOldEmitter", CreateOldEmitterPattern,
	"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx????" };
#endif

bool s_enabled = true;    // toggled by /classicfx
bool s_installed = false; // both hooks are in place
uintptr_t s_spawnCastAddress = 0;
uintptr_t s_createOldEmitterAddress = 0;

// The cast currently being spawned. Set by the spawner detour, read by the creator detour.
struct CastContext
{
	PlayerClient* caster = nullptr;
	PlayerClient* target = nullptr;
	const SpellEffect* record = nullptr; // spells.eff record of this cast; other records pass through untouched
};
CastContext s_cast;

// Sets the cast context for one SpawnCastEffect call and restores the previous one on every exit path.
class ScopedCastContext
{
public:
	ScopedCastContext(PlayerClient* caster, PlayerClient* target, const EQ_SpellExtra* effects)
		: m_previous(s_cast)
	{
		s_cast = { caster, target, effects ? effects->OldSpellEff : nullptr };
	}

	~ScopedCastContext() { s_cast = m_previous; }

	ScopedCastContext(const ScopedCastContext&) = delete;
	ScopedCastContext& operator=(const ScopedCastContext&) = delete;

private:
	CastContext m_previous;
};

// eqgame.exe's loaded image. g_eqgameimagesize is the end address, not a size.
uintptr_t ImageStart() { return reinterpret_cast<uintptr_t>(GetModuleHandleA(nullptr)); }
uintptr_t ImageEnd() { return g_eqgameimagesize; }

struct ScanResult
{
	uintptr_t address = 0; // first match
	int matches = 0;       // stops counting at 2
};

// Searches eqgame.exe for the signature's pattern.
ScanResult ScanImage(const Signature& sig)
{
	ScanResult result;
	uintptr_t imageStart = ImageStart();
	uintptr_t imageEnd = ImageEnd();
	size_t length = strlen(sig.mask);
	if (length == 0 || imageEnd < imageStart + length)
		return result;

	// FindPattern compares the whole mask at each start, so only search starts where the pattern fits.
	uintptr_t searchEnd = imageEnd - length + 1;
	for (uintptr_t address = imageStart; address < searchEnd && result.matches < 2; ++address)
	{
		address = FindPattern(address, searchEnd - address, sig.pattern, sig.mask);
		if (!address)
			break;
		if (result.matches++ == 0)
			result.address = address;
	}
	return result;
}

const char* DescribeScanFailure(const ScanResult& result)
{
	return result.matches == 0 ? "not found" : "ambiguous (several matches)";
}

#if !IS_ROF2_CLIENT
PlayerClient* ResolveActor(const SpellEffect* record, int section)
{
	// Same rule as the spawner, which compares case-insensitively: "Target" means the target if there is one,
	// anything else means the caster.
	const StageType& stage = record->stages[section];
	std::string_view tag = stage.AttachTag;
	if (s_cast.target && ci_equals(tag, "Target"))
		return s_cast.target;
	return s_cast.caster;
}

// Current world position of an attach node.
CVector3 GetNodePosition(CParticlePointInterface* node)
{
	const CVector4& position = node->GetWorldSpaceMatrix().row[3];
	return CVector3{ position.X, position.Y, position.Z };
}

// In the order the game stores it (Y, X, Z), which is the order the emitter position uses.
CVector3 GetActorPosition(PlayerClient* actor)
{
	return CVector3{ actor->Y, actor->X, actor->Z };
}

// Points the emitter at an attach node. From the next frame EQGraphics moves and orients it with that node,
// and removes it if the node is torn down.
void AttachEmitter(CParticleCloudInterface* handle, CParticlePointInterface* node)
{
	if (CEmitterInterface* emitter = handle->GetEmitterInterface())
		static_cast<CEmitter*>(emitter)->pAttachPoint = static_cast<CParticlePoint*>(node);
}
#else
// The hand emitter CreateOldEmitter_Detour just created for the current cast. The spawner asks this handle,
// and only this one, to attach itself next.
CParticleCloudInterface* s_handHandle = nullptr;

// CParticleCloud's vtable entry for slot 1 while it is replaced, and the stub that was there.
void** s_cloudAttachEntry = nullptr;
void* s_cloudAttachOriginal = nullptr;
bool s_cloudAttachWarned = false;

// Replaces CParticleCloudInterface::AttachToBone (thiscall, one argument; __fastcall with an unused edx matches
// that). For a hand emitter of the current cast it attaches the emitter to the bone the spawner chose. For
// anything else it does what the stub did.
int __fastcall AttachCloudToBone(CParticleCloudInterface* cloud, void* /*edx*/, CBoneInterface* bone)
{
	if (!s_enabled || !bone || !cloud || cloud != s_handHandle)
		return -1;
	s_handHandle = nullptr;

	CEmitterInterface* emitter = cloud->GetEmitterInterface();
	if (!emitter)
		return -1;

	static_cast<CEmitter*>(emitter)->pAttachBone = static_cast<CEQBone*>(bone);
	return 0;
}

bool WriteVtableEntry(void** entry, void* value)
{
	DWORD oldProtect = 0;
	if (!VirtualProtect(entry, sizeof(void*), PAGE_READWRITE, &oldProtect))
		return false;
	*entry = value;
	VirtualProtect(entry, sizeof(void*), oldProtect, &oldProtect);
	return true;
}

// Replaces CParticleCloud's attach stub, using the vtable of a handle the game just created. Only replaces it if
// the entry still points at the stub, so a different EQGraphicsDX9.dll is left alone.
bool HookCloudAttach(CParticleCloudInterface* handle)
{
	if (s_cloudAttachEntry)
		return true;

	void** entry = &(*reinterpret_cast<void***>(handle))[CloudAttachSlot];
	if (*entry == reinterpret_cast<void*>(&AttachCloudToBone))
		return true; // still ours from a load whose restore failed (see UnhookCloudAttach)
	if (memcmp(*entry, CloudAttachStub, sizeof(CloudAttachStub)) != 0)
		return false;

	void* original = *entry;
	if (!WriteVtableEntry(entry, reinterpret_cast<void*>(&AttachCloudToBone)))
		return false;

	s_cloudAttachEntry = entry;
	s_cloudAttachOriginal = original;
	return true;
}

// Puts the stub back. If that fails, the vtable still points into this DLL, so the DLL is pinned to stay loaded
// for the rest of the session. With no cast in progress AttachCloudToBone behaves exactly like the stub.
void UnhookCloudAttach()
{
	s_handHandle = nullptr;
	if (!s_cloudAttachEntry)
		return;

	if (WriteVtableEntry(s_cloudAttachEntry, s_cloudAttachOriginal))
	{
		s_cloudAttachEntry = nullptr;
		s_cloudAttachOriginal = nullptr;
		return;
	}

	HMODULE self = nullptr;
	bool pinned = GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
		reinterpret_cast<LPCSTR>(&AttachCloudToBone), &self) != FALSE;
	WriteChatf("\ay[ClassicSpellFX]\ax \arCould not restore EQGraphicsDX9.dll's attach stub; %s\ax",
		pinned ? "the plugin stays loaded until EverQuest restarts." : "pinning the plugin also failed.");
}
#endif

// Shrinks a hand emitter's spawn radius in the loaded spells.eff table and restores it when done.
// The creator reads the value during the call, so nothing is left pointing at modified data.
class ScopedHandRadius
{
public:
	ScopedHandRadius(SpellEffect* record, int section, int emitterIndex)
	{
		auto address = reinterpret_cast<uintptr_t>(&record->stages[section].Radius[emitterIndex]);

		// Only ever write inside eqgame's image, where the spells.eff table is loaded.
		if (address < ImageStart() || address + sizeof(float) > ImageEnd())
			return;

		m_radius = reinterpret_cast<float*>(address);
		m_original = *m_radius;
		if (m_original > 0.0f)
			*m_radius = m_original * HandSpawnRadiusScale;
	}

	~ScopedHandRadius()
	{
		if (m_radius)
			*m_radius = m_original;
	}

private:
	float* m_radius = nullptr;
	float m_original = 0.0f;
};

} // namespace

// Hook on eqgame's SpawnCastEffect. Arguments, in order: section, EQ_Spell*, EQ_SpellExtra* {OldSpellEff, NewSpellEff},
// caster, target, impact point (vec3 or null), extra, int.
// It only records the cast so CreateOldEmitter_Detour knows which actors and spells.eff record it belongs to.
DETOUR_TRAMPOLINE_DEF(int, SpawnCastEffect_Trampoline,
	(int, EQ_Spell*, EQ_SpellExtra*, PlayerClient*, PlayerClient*, const CVector3*, void*, int))
static int SpawnCastEffect_Detour(int section, EQ_Spell* spell, EQ_SpellExtra* effects, PlayerClient* caster,
	PlayerClient* target, const CVector3* impactPoint, void* extra, int arg8)
{
	// With spellsnew installed, a spell that has both records takes the new-effects path. Pass a copy of
	// the pair without the new record so it plays the classic effect; spells with no classic record keep theirs.
	EQ_SpellExtra classicOnly{};
	if (s_enabled && effects && effects->OldSpellEff && effects->NewSpellEff)
	{
		classicOnly.OldSpellEff = effects->OldSpellEff;
		effects = &classicOnly;
	}

	ScopedCastContext cast(caster, target, effects);
#if IS_ROF2_CLIENT
	int result = SpawnCastEffect_Trampoline(section, spell, effects, caster, target, impactPoint, extra, arg8);
	s_handHandle = nullptr; // only meaningful inside the cast
	return result;
#else
	return SpawnCastEffect_Trampoline(section, spell, effects, caster, target, impactPoint, extra, arg8);
#endif
}

// Hook on eqgame's CreateOldEmitter. Arguments, in order: spells.eff record, id, section, emitter index,
// mode (preset index), position (vec3), duration. Returns the emitter handle.
// Only emitters of the cast recorded above are changed; everything else goes straight to the original.
DETOUR_TRAMPOLINE_DEF(CParticleCloudInterface*, CreateOldEmitter_Trampoline, (SpellEffect*, int, int, int, int, const CVector3&, int))
static CParticleCloudInterface* CreateOldEmitter_Detour(SpellEffect* record, int id, int section, int emitterIndex, int mode,
	const CVector3& position, int duration)
{
	if (!s_enabled || !s_cast.caster || !record || record != s_cast.record
		|| section < 0 || section >= EffSectionCount || emitterIndex < 0 || emitterIndex >= EffEmittersPerSection)
	{
		return CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, position, duration);
	}

#if IS_ROF2_CLIENT
	// Hand effect: the spawner calls this once per hand and then asks the handle to attach to that hand, which
	// AttachCloudToBone answers.
	if (mode == ModeHands)
	{
		ScopedHandRadius handRadius(record, section, emitterIndex);
		CParticleCloudInterface* handle = CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode,
			position, duration);
		s_handHandle = nullptr;
		if (handle)
		{
			if (HookCloudAttach(handle))
				s_handHandle = handle;
			else if (!s_cloudAttachWarned)
			{
				s_cloudAttachWarned = true;
				WriteChatf("\ay[ClassicSpellFX]\ax \arEQGraphicsDX9.dll doesn't match: hand effects can't be attached.\ax");
			}
		}
		return handle;
	}
#else
	// Hand effect: create one emitter per hand, each starting at and attached to that hand.
	if (mode == ModeHands)
	{
		ScopedHandRadius handRadius(record, section, emitterIndex);
		PlayerClient* actor = ResolveActor(record, section);
		CParticleCloudInterface* firstHandle = nullptr;
		bool foundHand = false;

		for (int slot : AttachSlotHands)
		{
			CParticlePointInterface* node = actor->mActorClient.pParticlePoints[slot];
			if (!node)
				continue;

			foundHand = true;
			CParticleCloudInterface* handle = CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode,
				GetNodePosition(node), duration);
			if (handle)
			{
				AttachEmitter(handle, node);
				if (!firstHandle)
					firstHandle = handle;
			}
		}

		// The spawner ignores the returned handle, so returning one of the two is enough.
		if (foundHand)
			return firstHandle;

		// No hand nodes on this model: put it on the actor rather than the world origin.
		return CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, GetActorPosition(actor), duration);
	}
#endif

	return CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, position, duration);
}

// /classicfx [on|off]; with no argument it shows the current state.
static void ClassicSpellFXCommand(PlayerClient*, const char* line)
{
	char arg[MAX_STRING] = { 0 };
	GetArg(arg, line, 1);

	if (ci_equals(arg, "on"))
		s_enabled = true;
	else if (ci_equals(arg, "off"))
		s_enabled = false;
	else if (arg[0])
	{
		WriteChatf("\ay[ClassicSpellFX]\ax Usage: /classicfx [on|off]");
		return;
	}

	if (!s_installed)
		WriteChatf("\ay[ClassicSpellFX]\ax \arNot installed (signature check failed at load).\ax");
	else
		WriteChatf("\ay[ClassicSpellFX]\ax %s.", s_enabled ? "On" : "Off");
}

PLUGIN_API void InitializePlugin()
{
	DebugSpewAlways("Initializing MQ2ClassicSpellFX");

	AddCommand("/classicfx", ClassicSpellFXCommand, false, true, true);

	bool ok = true;

	// The two hook targets must each match exactly once.
	for (auto [sig, address] : { std::pair{ &SpawnCastSig, &s_spawnCastAddress },
		std::pair{ &CreateOldEmitterSig, &s_createOldEmitterAddress } })
	{
		ScanResult result = ScanImage(*sig);
		*address = result.matches == 1 ? result.address : 0;
		if (!*address)
		{
			WriteChatf("\ay[ClassicSpellFX]\ax \arSignature failed: %s (%s)\ax", sig->name, DescribeScanFailure(result));
			ok = false;
		}
	}

	if (!ok)
	{
		WriteChatf("\ay[ClassicSpellFX]\ax \arCould not find the functions to hook, ClassicSpellFX will be disabled.\ax");
		return;
	}

	// Hook the creator first: on its own it changes nothing, because only the spawner hook sets the cast context.
	if (!EzDetour(s_createOldEmitterAddress, CreateOldEmitter_Detour, CreateOldEmitter_Trampoline))
	{
		WriteChatf("\ay[ClassicSpellFX]\ax \arDisabled: could not hook %s\ax", CreateOldEmitterSig.name);
		return;
	}
	if (!EzDetour(s_spawnCastAddress, SpawnCastEffect_Detour, SpawnCastEffect_Trampoline))
	{
		RemoveDetour(s_createOldEmitterAddress);
		WriteChatf("\ay[ClassicSpellFX]\ax \arDisabled: could not hook %s\ax", SpawnCastSig.name);
		return;
	}
	s_installed = true;

	WriteChatf("\ay[ClassicSpellFX]\ax Installed. /classicfx on|off toggles it.");
}

PLUGIN_API void ShutdownPlugin()
{
	DebugSpewAlways("Shutting down MQ2ClassicSpellFX");

	RemoveCommand("/classicfx");

	if (s_installed)
	{
		RemoveDetour(s_spawnCastAddress);
		RemoveDetour(s_createOldEmitterAddress);
#if IS_ROF2_CLIENT
		UnhookCloudAttach();
#endif
		s_installed = false;
	}
}
