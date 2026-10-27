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
// Every address and field offset is checked with a byte pattern at load. If any check fails the
// plugin stays inactive.

#include <mq/Plugin.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

PreSetup("MQ2ClassicSpellFX");
PLUGIN_VERSION(0.1);

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
// that signature's comment, confirm the offsets its comment lists, then rebuild the pattern from the new
// bytes (wildcard RIP-relative displacements with ??) and check it is unique in the module's .text.

constexpr size_t EffRecordHeaderSize = 0x8;
constexpr size_t EffSectionSize = 0x3A0;
constexpr size_t EffSectionTagOffset = 0x60;
constexpr size_t EffEmitterRadiusOffset = 0x25C; // float per emitter, section-relative

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
// ActorPositionOffset: the spawner's code-2 case loads the actor's position as three floats:
//   movss xmm, [rdi+0x74] / [rdi+0x78] / [rdi+0x7C]
// The new-system spawner (see below) loads the same three floats from its caster register.
//
// ActorAttachNodesOffset: a tiny getter (was 0x140311740), checked by the "GetAttachNode" signature:
//   movsxd rax, edx
//   mov    rax, [rcx + rax*8 + 0x1120]
//   ret
// Find it from the new-system spawner (was 0x1402882f0, called by SpawnCastEffect when the spell has a
// spellsnew record): its origin switch has a row of `mov edx, N / mov rcx, rbx / call <getter>` for
// N = 1..9, falling back to `xor edx, edx / call <getter>` when the result is null. That getter's
// displacement is the offset. If the getter is inlined in a new build, the same displacement appears
// in those switch cases instead.
//
// AttachSlotHands: in that switch, new-system origin 04 passes slot 3 and origin 05 passes slot 4. The
// model setup function (it references "%sR_POINT_DAG" and "%sL_POINT_DAG") binds slot 3 to R_POINT_DAG
// and slot 4 to L_POINT_DAG (look for `mov edx, 3` / `mov edx, 4` before a virtual call that takes the
// formatted name). The P99 community labels origin 04 as the left hand, so R/L may be mirrored; it
// doesn't matter because both hands are used.
constexpr size_t ActorPositionOffset = 0x74;
constexpr size_t ActorAttachNodesOffset = 0x1120;
constexpr int AttachSlotHands[] = { 3, 4 }; // R_POINT_DAG, L_POINT_DAG

//----------------------------------------------------------------------------
// EQGraphics.dll objects
//----------------------------------------------------------------------------
// How to get into the old particle code: CreateGraphicsEngine (the DLL's only interesting export) fills
// a struct of interface pointers. Its +0x10 member is what eqgame stores in the global that
// CreateOldEmitter calls through (`call [rax+0xF8]`). That object's vtable entry 0xF8 is the old
// CreateEmitter (was 0x18009d200). It takes (name, sprite, params, pos*, out**) plus two trailing
// arguments it never reads.
//
// EmitterHandleEmitterOffset: at the end of the old CreateEmitter the returned handle is allocated as
// 0x18 bytes (`mov ecx, 0x18 / call alloc`) and filled with: vtable, `mov dword [rcx+8], 2`, and the
// emitter pointer at +0x10. Checked by the "EmitterHandle" signature. CreateOldEmitter returns this
// handle to the spawner (which ignores it).
//
// EmitterAttachNodeOffset: the emitter (CEmitter) constructor (was 0x18009ee00) is called from
// CEmitterDef::CreateEmitter right after a 0xC8-byte allocation. The xref to the string
// "CEmitter construction failed." is immediately after that call. The constructor copies stack
// arguments into the object: model -> +0x70, another attach object -> +0x78, attach node -> +0x80.
// Checked by "EmitterCtorNode". To confirm the meaning, find the per-emitter update: the function
// called 4 times from the function that references "CParticleSystem::ProcessFrame" (one call per emitter
// list). In it, the position source is chosen in this order: +0x78, then +0x70 (model), then +0x80:
//   mov rcx, [rdi+0x80] / test rcx, rcx / je / mov rax, [rcx] / call [rax+8]
// and then the result's +0x30/+0x34/+0x38 are copied to the emitter's position (+0x3C..+0x44).
// Checked by "EmitterUpdateNode". A little later, under `cmp eax, 1` (def+0x60, orient-to-attach mode),
// the same node's matrix rows are copied into the emitter orientation (+0x48..+0x68). Also,
// RemoveEmittersForNode (attach-point teardown, called for every list including the old one) has
// `cmp [rbx+0x80], r13`. That is what keeps attached emitters from outliving their node.
//
// NodeMatrixPositionIndex / vtable slot 1: the attach node's vtable[1] (`call [rax+8]` above) refreshes
// the node from its bone and returns `lea rax, [node+0x10]`: a 4x4 float world matrix with the
// translation at byte +0x30 = float index 12. eqgame's new-system spawner uses the same call and reads
// +0x30/+0x34/+0x38 of the result, which is a second confirmation.
constexpr size_t EmitterHandleEmitterOffset = 0x10;
constexpr size_t EmitterAttachNodeOffset = 0x80;
constexpr size_t NodeMatrixPositionIndex = 12; // float index of the translation row in the world matrix

struct Signature
{
	const char* name;
	const wchar_t* module;
	const char* pattern;
};

// SpawnCastEffect (eqgame.exe, was 0x140287be0): spawns a spell's effect for one section (cast, middle,
// landing).
//   int SpawnCastEffect(int section, Spell*, EffectPair*, Actor* caster, Actor* target,
//                       const float* impactPoint, void* extra, int)
// How to find it:
//  - EffectPair is {oldRecord*, newRecord*}. The spawner checks the new pointer first and calls the
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
const Signature SpawnCastSig = { "SpawnCastEffect", L"eqgame.exe",
	"40 55 53 57 41 54 41 57 48 8D 6C 24 E9 48 81 EC E0 00 00 00 4C 63 F9 4D 8B E1 49 8B D8 48 8B FA 4D 85 C9 0F 84" };

// CreateOldEmitter (eqgame.exe, was 0x1402911d0): creates one old-style emitter.
//   EmitterHandle* CreateOldEmitter(const EffRecord*, int id, int section, int emitterIndex,
//                                   int mode, const float* position, int duration)
// How to find it:
//  - It is the only code that references the format string "%s_%d_%d_%d_%d" (the emitter name).
//  - It indexes the 7-entry preset table with the mode: `movsxd rsi, [rbp+0x150]` then
//    `mov r9, [table + rsi*8]`. The trailing "48 63 B5 50 01 00 00" in the pattern is that mode load,
//    so if the stack layout changes the pattern fails rather than reading the wrong argument.
//  - It does `imul rcx, rdi, 0x3A0`, reads the per-emitter overrides at [r8 + r14*4 + 0x22C..0x298],
//    and ends with a virtual call `call [rax+0xF8]` on the old particle object (see EQGraphics notes).
//  - Both spawners call it at the end of every location-code case.
const Signature CreateOldEmitterSig = { "CreateOldEmitter", L"eqgame.exe",
	"40 55 56 57 41 54 41 56 41 57 48 8D AC 24 08 FF FF FF 48 81 EC F8 01 00 00 48 8B 05 ?? ?? ?? ?? 48 33 C4 48 89 85 E0 00 00 00 48 63 B5 50 01 00 00" };

// Checks only: these confirm the field offsets above are still what the code uses. They don't locate
// anything the plugin calls. If one fails, re-derive the offset with the notes above, update the
// constant and the literal offset bytes in the pattern together.
const Signature OffsetChecks[] = {
	// actor->attachNodes[slot] lives at +0x1120 (the "20 11 00 00" bytes are ActorAttachNodesOffset)
	{ "GetAttachNode", L"eqgame.exe", "48 63 C2 48 8B 84 C1 20 11 00 00 C3" },
	// emitter handle stores the emitter at +0x10 ("48 89 69 10" / "48 89 41 10" are EmitterHandleEmitterOffset)
	{ "EmitterHandle", L"EQGraphics.dll",
		"48 8D 05 ?? ?? ?? ?? C7 41 08 02 00 00 00 48 89 01 48 89 69 10 EB 03 48 8B CD 48 8B 44 24 60 48 89 41 10" },
	// emitter constructor stores the attach node at +0x80 ("48 89 83 80 00 00 00" is EmitterAttachNodeOffset)
	{ "EmitterCtorNode", L"EQGraphics.dll", "48 8B 84 24 90 00 00 00 48 89 83 80 00 00 00 48 89 BB 88 00 00 00" },
	// per-frame update follows the node at +0x80 and calls its GetWorldMatrix (vtable slot 1, "FF 50 08")
	{ "EmitterUpdateNode", L"EQGraphics.dll",
		"48 8B 8F 80 00 00 00 48 85 C9 0F 84 ?? ?? ?? ?? 48 8B 01 FF 50 08 48 8B 8F 80 00 00 00" },
};

bool s_enabled = true;
bool s_installed = false;
uintptr_t s_spawnCastAddress = 0;
uintptr_t s_createOldEmitterAddress = 0;

// The cast currently being spawned. Set by the spawner detour, read by the creator detour.
struct CastContext
{
	uint8_t* caster = nullptr;
	uint8_t* target = nullptr;
	const float* impactPoint = nullptr;
	bool active = false;
};
CastContext s_cast;

// Holds the cast context for one SpawnCastEffect call and restores the previous one on every exit
// path, so the creator detour never sees stale actor pointers with `active` still set.
class ScopedCastContext
{
public:
	ScopedCastContext(void* caster, void* target, const float* impactPoint)
		: m_previous(s_cast)
	{
		s_cast = { static_cast<uint8_t*>(caster), static_cast<uint8_t*>(target), impactPoint, true };
	}

	~ScopedCastContext() { s_cast = m_previous; }

	ScopedCastContext(const ScopedCastContext&) = delete;
	ScopedCastContext& operator=(const ScopedCastContext&) = delete;

private:
	CastContext m_previous;
};

bool ParsePattern(const char* text, std::vector<uint8_t>& bytes, std::vector<bool>& mask)
{
	for (const char* p = text; *p;)
	{
		if (*p == ' ') { ++p; continue; }
		if (p[0] == '?' && p[1] == '?')
		{
			bytes.push_back(0);
			mask.push_back(false);
			p += 2;
			continue;
		}

		char hex[3] = { p[0], p[1], 0 };
		char* end = nullptr;
		unsigned long value = strtoul(hex, &end, 16);
		if (end != hex + 2)
			return false;

		bytes.push_back(static_cast<uint8_t>(value));
		mask.push_back(true);
		p += 2;
	}
	return !bytes.empty();
}

// Returns the address of the only match in the module's .text section, or 0 if there are zero or several.
uintptr_t FindUnique(const Signature& sig)
{
	HMODULE module = GetModuleHandleW(sig.module);
	if (!module)
		return 0;

	auto base = reinterpret_cast<uint8_t*>(module);
	auto nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
	IMAGE_SECTION_HEADER* section = IMAGE_FIRST_SECTION(nt);

	const uint8_t* text = nullptr;
	size_t textSize = 0;
	for (int i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section)
	{
		// Full 8-byte compare so ".textbss" or similar can't be taken for ".text".
		if (memcmp(section->Name, ".text\0\0\0", IMAGE_SIZEOF_SHORT_NAME) == 0)
		{
			text = base + section->VirtualAddress;
			textSize = section->Misc.VirtualSize;
			break;
		}
	}

	std::vector<uint8_t> bytes;
	std::vector<bool> mask;
	if (!text || !ParsePattern(sig.pattern, bytes, mask) || textSize < bytes.size())
		return 0;

	uintptr_t found = 0;
	for (size_t i = 0; i <= textSize - bytes.size(); ++i)
	{
		// Cheap first-byte reject, but only when byte 0 isn't a wildcard.
		if (mask[0] && text[i] != bytes[0])
			continue;

		size_t j = 1;
		while (j < bytes.size() && (!mask[j] || text[i + j] == bytes[j]))
			++j;

		if (j == bytes.size())
		{
			if (found)
				return 0;
			found = reinterpret_cast<uintptr_t>(text + i);
		}
	}
	return found;
}

uint8_t* ResolveActor(const uint8_t* record, int section)
{
	// Same rule as the spawner: "Target" means the target if there is one, anything else means the caster.
	auto tag = reinterpret_cast<const char*>(record + EffRecordHeaderSize + section * EffSectionSize + EffSectionTagOffset);
	if (s_cast.target && strncmp(tag, "Target", 32) == 0)
		return s_cast.target;
	return s_cast.caster;
}

void* GetAttachNode(uint8_t* actor, int slot)
{
	return *reinterpret_cast<void**>(actor + ActorAttachNodesOffset + slot * sizeof(void*));
}

void GetNodePosition(void* node, float (&pos)[3])
{
	using GetWorldMatrixFn = float* (*)(void*);
	GetWorldMatrixFn getWorldMatrix = (*reinterpret_cast<GetWorldMatrixFn**>(node))[1];
	const float* matrix = getWorldMatrix(node);
	pos[0] = matrix[NodeMatrixPositionIndex];
	pos[1] = matrix[NodeMatrixPositionIndex + 1];
	pos[2] = matrix[NodeMatrixPositionIndex + 2];
}

void GetActorPosition(uint8_t* actor, float (&pos)[3])
{
	memcpy(pos, actor + ActorPositionOffset, sizeof(pos));
}

void AttachEmitter(void* handle, void* node)
{
	auto emitter = *reinterpret_cast<uint8_t**>(static_cast<uint8_t*>(handle) + EmitterHandleEmitterOffset);
	if (emitter)
		*reinterpret_cast<void**>(emitter + EmitterAttachNodeOffset) = node;
}

// Shrinks a hand emitter's spawn radius in the loaded spells.eff table and restores it when done.
// The creator reads the value during the call, so nothing is left pointing at modified data.
class ScopedHandRadius
{
public:
	ScopedHandRadius(const uint8_t* record, int section, int emitterIndex)
		: m_radius(reinterpret_cast<float*>(const_cast<uint8_t*>(record) + EffRecordHeaderSize
			+ section * EffSectionSize + EffEmitterRadiusOffset + emitterIndex * sizeof(float)))
		, m_original(*m_radius)
	{
		if (m_original > 0.0f)
			*m_radius = m_original * HandSpawnRadiusScale;
	}

	~ScopedHandRadius() { *m_radius = m_original; }

private:
	float* m_radius;
	float m_original;
};

} // namespace

// int SpawnCastEffect(int section, Spell* spell, EffectPair* effects, Actor* caster, Actor* target,
//     const float* impactPoint, void* extra, int arg8)
DETOUR_TRAMPOLINE_DEF(int, SpawnCastEffect_Trampoline, (int, void*, void*, void*, void*, const float*, void*, int))
static int SpawnCastEffect_Detour(int section, void* spell, void* effects, void* caster, void* target,
	const float* impactPoint, void* extra, int arg8)
{
	ScopedCastContext cast(caster, target, impactPoint);
	return SpawnCastEffect_Trampoline(section, spell, effects, caster, target, impactPoint, extra, arg8);
}

// EmitterHandle* CreateOldEmitter(const EffRecord* record, int id, int section, int emitterIndex,
//     int mode, const float* position, int duration)
DETOUR_TRAMPOLINE_DEF(void*, CreateOldEmitter_Trampoline, (const uint8_t*, int, int, int, int, const float*, int))
static void* CreateOldEmitter_Detour(const uint8_t* record, int id, int section, int emitterIndex, int mode,
	const float* position, int duration)
{
	if (!s_enabled || !s_cast.active || !s_cast.caster || !record)
		return CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, position, duration);

	if (mode == ModeHands)
	{
		uint8_t* actor = ResolveActor(record, section);
		void* firstHandle = nullptr;
		bool foundHand = false;

		{
			// Only emitters that actually sit on a hand get the shrunk disc; the actor-position
			// fallback below must keep the body-sized radius.
			ScopedHandRadius handRadius(record, section, emitterIndex);

			for (int slot : AttachSlotHands)
			{
				void* node = GetAttachNode(actor, slot);
				if (!node)
					continue;

				foundHand = true;
				float handPos[3];
				GetNodePosition(node, handPos);

				void* handle = CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, handPos, duration);
				if (handle)
				{
					AttachEmitter(handle, node);
					if (!firstHandle)
						firstHandle = handle;
				}
			}
		}

		if (foundHand)
			return firstHandle;

		// No hand nodes on this model: put it on the actor rather than the world origin.
		float actorPos[3];
		GetActorPosition(actor, actorPos);
		return CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, actorPos, duration);
	}

	if (mode == ModeImpactPoint && !s_cast.impactPoint)
	{
		float actorPos[3];
		GetActorPosition(ResolveActor(record, section), actorPos);
		return CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, actorPos, duration);
	}

	return CreateOldEmitter_Trampoline(record, id, section, emitterIndex, mode, position, duration);
}

static void ClassicSpellFXCommand(PlayerClient*, const char* line)
{
	char arg[MAX_STRING] = { 0 };
	GetArg(arg, line, 1);

	if (ci_equals(arg, "on"))
		s_enabled = true;
	else if (ci_equals(arg, "off"))
		s_enabled = false;
	else if (!arg[0])
		s_enabled = !s_enabled;
	else
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

	AddCommand("/classicfx", ClassicSpellFXCommand);

	bool ok = true;
	for (const Signature& check : OffsetChecks)
	{
		if (!FindUnique(check))
		{
			WriteChatf("\ay[ClassicSpellFX]\ax \arSignature check failed: %s\ax", check.name);
			ok = false;
		}
	}

	s_spawnCastAddress = FindUnique(SpawnCastSig);
	s_createOldEmitterAddress = FindUnique(CreateOldEmitterSig);
	if (!s_spawnCastAddress)
		WriteChatf("\ay[ClassicSpellFX]\ax \arSignature not found: %s\ax", SpawnCastSig.name);
	if (!s_createOldEmitterAddress)
		WriteChatf("\ay[ClassicSpellFX]\ax \arSignature not found: %s\ax", CreateOldEmitterSig.name);

	if (!ok || !s_spawnCastAddress || !s_createOldEmitterAddress)
	{
		WriteChatf("\ay[ClassicSpellFX]\ax \arDisabled: the client changed, offsets need re-checking.\ax");
		return;
	}

	EzDetour(s_createOldEmitterAddress, CreateOldEmitter_Detour, CreateOldEmitter_Trampoline);
	EzDetour(s_spawnCastAddress, SpawnCastEffect_Detour, SpawnCastEffect_Trampoline);
	s_installed = true;

	WriteChatf("\ay[ClassicSpellFX]\ax Installed. /classicfx toggles it on and off.");
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
