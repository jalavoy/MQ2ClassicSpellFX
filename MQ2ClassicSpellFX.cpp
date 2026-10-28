// MQ2ClassicSpellFX - puts the classic (spells.eff) hand effects back on the caster's hands.
//
// This plugin hooks the spell-cast effect spawner and the old-style emitter creator:
//  - code 0: one emitter per hand, attached to the hand's attach node. EQGraphics then moves the
//    emitter with the bone and orients it with the bone every frame (old-style emitter
//    definitions always use orient-to-attach mode), so the spray follows the forearm.
//    A few hand emitters have a spawn radius (a disc around the emitter) sized for the body;
//    it is scaled down so the disc stays inside the hand.
//  - code 1 without an impact point: the emitter is placed at the actor instead of 0,0,0.
//
// Only the cast spawner is hooked. The persistent object-effect spawner keeps its emitter handle,
// and attached emitters are deleted by EQGraphics when the actor's nodes are torn down, so
// attaching there would leave that handle dangling.
//
// Spell effect records, actor fields and the EQGraphics emitter types come from eqlib. The two hooked
// functions are found with a byte pattern at load. If either isn't found exactly once the plugin stays inactive.

#include <mq/Plugin.h>

#include <cstdint>
#include <cstring>
#include <iterator>
#include <string_view>
#include <type_traits>
#include <utility>

PreSetup("MQ2ClassicSpellFX");
PLUGIN_VERSION(0.6);

namespace {

//============================================================================
// Re-finding these values after a patch
//============================================================================
//
// Everything below was derived from the 2026-10-19 live client (eqgame.exe and EQGraphics.dll of that
// date). Addresses quoted as "was 0x14..." / "was 0x18..." are from that build at the preferred image
// base; they WILL move, so use them only to orient yourself in an old disassembly. The anchors that
// survive patches are strings, distinctive constants (0xAE8, 0x3A0) and the call relationships
// described here.
//
// Tools that worked:
//  - Ghidra (string search, xrefs, disassembly). The decompiler may fail on this binary; disassembly is enough.
//  - objdump -d -M intel -j .text eqgame.exe > text.asm, then grep. Fast for "find every imul by 0x3A0".
//
// When a signature fails at load, the plugin prints its name. Find the function again with the steps in
// that signature's comment, then rebuild the byte array and mask from the new bytes (wildcard RIP-relative
// displacements: 0x00 in the array, '?' in the mask) and check it is unique in the module's .text.

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
constexpr int ModeHands = 0;         // location code 0
constexpr int ModeImpactPoint = 5;   // location code 1

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
// CreateOldEmitter ends in the old particle object's CreateEmitter (`call [rax+0xF8]`, was 0x18009d200),
// which goes through CParticleSystem::CreateSpellEmitter. That path always creates the emitter without an
// attach point, so the plugin sets CEmitter::pAttachPoint itself afterwards.

struct Signature
{
	const char* name;
	const wchar_t* module;
	const uint8_t* pattern;
	const char* mask; // eqlib FindPattern mask: 'x' = match, '?' = wildcard
};

// SpawnCastEffect (eqgame.exe, was 0x140287be0): spawns a spell's effect for one section (cast, middle,
// landing).
//   int SpawnCastEffect(int section, EQ_Spell*, EQ_SpellExtra*, PlayerClient* caster, PlayerClient* target,
//                       const float* impactPoint, void* extra, int)
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
const Signature SpawnCastSig = { "SpawnCastEffect", L"eqgame.exe", SpawnCastPattern,
	"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" };

// CreateOldEmitter (eqgame.exe, was 0x1402911d0): creates one old-style emitter.
//   CEmitterInterface* CreateOldEmitter(SpellEffect*, int id, int section, int emitterIndex,
//                                   int mode, const float* position, int duration)
// How to find it:
//  - It is the only code that references the format string "%s_%d_%d_%d_%d" (the emitter name).
//  - It indexes the 7-entry preset table with the mode: `movsxd rsi, [rbp+0x150]` then
//    `mov r9, [table + rsi*8]`. The trailing "48 63 B5 50 01 00 00" in the pattern is that mode load,
//    so if the stack layout changes the pattern fails rather than reading the wrong argument.
//  - It does `imul rcx, rdi, 0x3A0`, reads the per-emitter overrides at [r8 + r14*4 + 0x22C..0x298],
//    and ends with a virtual call `call [rax+0xF8]` on the old particle object (see the EQGraphics note).
//  - Both spawners call it at the end of every location-code case.
const uint8_t CreateOldEmitterPattern[] = {
	0x40, 0x55, 0x56, 0x57, 0x41, 0x54, 0x41, 0x56, 0x41, 0x57, 0x48, 0x8D,
	0xAC, 0x24, 0x08, 0xFF, 0xFF, 0xFF, 0x48, 0x81, 0xEC, 0xF8, 0x01, 0x00,
	0x00, 0x48, 0x8B, 0x05, 0x00, 0x00, 0x00, 0x00, 0x48, 0x33, 0xC4, 0x48,
	0x89, 0x85, 0xE0, 0x00, 0x00, 0x00, 0x48, 0x63, 0xB5, 0x50, 0x01, 0x00,
	0x00
};
const Signature CreateOldEmitterSig = { "CreateOldEmitter", L"eqgame.exe", CreateOldEmitterPattern,
	"xxxxxxxxxxxxxxxxxxxxxxxxxxxx????xxxxxxxxxxxxxxxxx" };

bool s_enabled = true;    // toggled by /classicfx
bool s_installed = false; // both hooks are in place
uintptr_t s_spawnCastAddress = 0;
uintptr_t s_createOldEmitterAddress = 0;

// eqgame.exe's .data section, where the spells.eff table lives (the only memory the plugin writes to).
uintptr_t s_dataStart = 0;
uintptr_t s_dataEnd = 0;

// The cast currently being spawned. Set by the spawner detour, read by the creator detour.
struct CastContext
{
	PlayerClient* caster = nullptr;
	PlayerClient* target = nullptr;
	const float* impactPoint = nullptr;
	const SpellEffect* record = nullptr; // spells.eff record of this cast; other records pass through untouched
};
CastContext s_cast;

// Sets the cast context for one SpawnCastEffect call and restores the previous one on every exit path.
class ScopedCastContext
{
public:
	ScopedCastContext(PlayerClient* caster, PlayerClient* target, const float* impactPoint, const EQ_SpellExtra* effects)
		: m_previous(s_cast)
	{
		s_cast = { caster, target, impactPoint, effects ? effects->OldSpellEff : nullptr };
	}

	~ScopedCastContext() { s_cast = m_previous; }

	ScopedCastContext(const ScopedCastContext&) = delete;
	ScopedCastContext& operator=(const ScopedCastContext&) = delete;

private:
	CastContext m_previous;
};

// Finds a section of a loaded module by its exact name. Returns false if the module or section is missing.
bool GetModuleSection(const wchar_t* moduleName, const char* sectionName, uintptr_t& start, size_t& size)
{
	HMODULE module = GetModuleHandleW(moduleName);
	if (!module)
		return false;

	auto base = reinterpret_cast<uint8_t*>(module);
	auto nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
	IMAGE_SECTION_HEADER* section = IMAGE_FIRST_SECTION(nt);

	for (int i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section)
	{
		if (strncmp(reinterpret_cast<const char*>(section->Name), sectionName, IMAGE_SIZEOF_SHORT_NAME) == 0)
		{
			start = reinterpret_cast<uintptr_t>(base + section->VirtualAddress);
			size = section->Misc.VirtualSize;
			return true;
		}
	}
	return false;
}

struct ScanResult
{
	uintptr_t address = 0; // first match
	int matches = 0;       // stops counting at 2
};

// Searches the signature's module .text for its pattern.
ScanResult ScanText(const Signature& sig)
{
	ScanResult result;
	uintptr_t textStart = 0;
	size_t textSize = 0;
	size_t length = strlen(sig.mask);
	if (!GetModuleSection(sig.module, ".text", textStart, textSize) || length == 0 || textSize < length)
		return result;

	// FindPattern compares the whole mask at each start, so only search starts where the pattern fits.
	uintptr_t searchEnd = textStart + textSize - length + 1;
	for (uintptr_t address = textStart; address < searchEnd && result.matches < 2; ++address)
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

PlayerClient* ResolveActor(const SpellEffect* record, int section)
{
	// Same rule as the spawner, which compares case-insensitively: "Target" means the target if there is one,
	// anything else means the caster.
	const StageType& stage = record->stages[section];
	std::string_view tag(stage.AttachTag, strnlen(stage.AttachTag, std::size(stage.AttachTag)));
	if (s_cast.target && ci_equals(tag, "Target"))
		return s_cast.target;
	return s_cast.caster;
}

// Current world position of an attach node.
void GetNodePosition(CParticlePointInterface* node, float (&pos)[3])
{
	const CVector4& position = node->GetWorldMatrix()->row[3];
	pos[0] = position.X;
	pos[1] = position.Y;
	pos[2] = position.Z;
}

// In the order the game stores it (Y, X, Z), which is the order the emitter position uses.
void GetActorPosition(PlayerClient* actor, float (&pos)[3])
{
	pos[0] = actor->Y;
	pos[1] = actor->X;
	pos[2] = actor->Z;
}

// Points the emitter at an attach node. From the next frame EQGraphics moves and orients it with that node,
// and removes it if the node is torn down.
void AttachEmitter(CEmitterInterface* handle, CParticlePointInterface* node)
{
	if (CEmitter* emitter = handle->GetEmitter())
		emitter->pAttachPoint = node;
}

// Shrinks a hand emitter's spawn radius in the loaded spells.eff table and restores it when done.
// The creator reads the value during the call, so nothing is left pointing at modified data.
class ScopedHandRadius
{
public:
	ScopedHandRadius(SpellEffect* record, int section, int emitterIndex)
	{
		auto address = reinterpret_cast<uintptr_t>(&record->stages[section].Radius[emitterIndex]);

		// Only ever write inside eqgame's writable .data, where the spells.eff table is loaded.
		if (address < s_dataStart || address + sizeof(float) > s_dataEnd)
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
	(int, EQ_Spell*, EQ_SpellExtra*, PlayerClient*, PlayerClient*, const float*, void*, int))
static int SpawnCastEffect_Detour(int section, EQ_Spell* spell, EQ_SpellExtra* effects, PlayerClient* caster,
	PlayerClient* target, const float* impactPoint, void* extra, int arg8)
{
	// With spellsnew installed, a spell that has both records takes the new-effects path. Pass a copy of
	// the pair without the new record so it plays the classic effect; spells with no classic record keep theirs.
	EQ_SpellExtra classicOnly{};
	if (s_enabled && effects && effects->OldSpellEff && effects->NewSpellEff)
	{
		classicOnly.OldSpellEff = effects->OldSpellEff;
		effects = &classicOnly;
	}

	ScopedCastContext cast(caster, target, impactPoint, effects);
	return SpawnCastEffect_Trampoline(section, spell, effects, caster, target, impactPoint, extra, arg8);
}

// Hook on eqgame's CreateOldEmitter. Arguments, in order: spells.eff record, id, section, emitter index,
// mode (preset index), position (vec3), duration. Returns the emitter handle.
// Only emitters of the cast recorded above are changed; everything else goes straight to the original.
DETOUR_TRAMPOLINE_DEF(CEmitterInterface*, CreateOldEmitter_Trampoline, (SpellEffect*, int, int, int, int, const float*, int))
static CEmitterInterface* CreateOldEmitter_Detour(SpellEffect* record, int id, int section, int emitterIndex, int mode,
	const float* position, int duration)
{
	if (!s_enabled || !s_cast.caster || !record || record != s_cast.record
		|| section < 0 || section >= EffSectionCount || emitterIndex < 0 || emitterIndex >= EffEmittersPerSection)
	{
		return CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, position, duration);
	}

	// Hand effect: create one emitter per hand, each starting at and attached to that hand.
	if (mode == ModeHands)
	{
		ScopedHandRadius handRadius(record, section, emitterIndex);
		PlayerClient* actor = ResolveActor(record, section);
		CEmitterInterface* firstHandle = nullptr;
		bool foundHand = false;

		for (int slot : AttachSlotHands)
		{
			CParticlePointInterface* node = actor->mActorClient.pParticlePoints[slot];
			if (!node)
				continue;

			foundHand = true;
			float handPos[3];
			GetNodePosition(node, handPos);

			CEmitterInterface* handle = CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, handPos, duration);
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
		float actorPos[3];
		GetActorPosition(actor, actorPos);
		return CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, actorPos, duration);
	}

	// Impact-point effect with no impact point (most single-target spells): use the actor instead of 0,0,0.
	if (mode == ModeImpactPoint && !s_cast.impactPoint)
	{
		float actorPos[3];
		GetActorPosition(ResolveActor(record, section), actorPos);
		return CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, actorPos, duration);
	}

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
		ScanResult result = ScanText(*sig);
		*address = result.matches == 1 ? result.address : 0;
		if (!*address)
		{
			WriteChatf("\ay[ClassicSpellFX]\ax \arSignature failed: %s (%s)\ax", sig->name, DescribeScanFailure(result));
			ok = false;
		}
	}

	// Bounds for ScopedHandRadius's write into the spells.eff table.
	size_t dataSize = 0;
	if (!GetModuleSection(L"eqgame.exe", ".data", s_dataStart, dataSize))
	{
		WriteChatf("\ay[ClassicSpellFX]\ax \arCould not find eqgame.exe's .data section\ax");
		ok = false;
	}
	s_dataEnd = s_dataStart + dataSize;

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
		s_installed = false;
	}
}
