# InfernoSaber context

InfernoSaber is a Beat Saber automapper that generates custom levels from song
audio, with adjustable difficulty and learned mapping styles. This glossary
provides shared language for discussing its behavior and future improvements.

## Language

### Songs and maps

**Song**:
An audio recording, together with descriptive metadata such as its title and
artist, used as the basis for a custom level.

**Beatmap**:
The timed gameplay objects and events for one song, gameplay characteristic, and
difficulty variant.

**Map package**:
A playable custom level containing song audio, level metadata, one or more
beatmaps, and supporting assets such as a cover image. A ZIP is one way to
distribute a map package.
_Avoid_: Using "map" where the distinction between a beatmap and its package matters.

**Difficulty setting**:
InfernoSaber's BPS-based control for the approximate challenge of generated
gameplay, including movement and note density. It is a generation input, distinct
from measured BPS or an external star rating.

**Difficulty label**:
A Beat Saber difficulty name, such as Expert or ExpertPlus, assigned to a beatmap
within a map package. It identifies a variant without fully describing its challenge.

**Mapping style**:
The gameplay patterns and placement preferences learned from a selection of
human-authored beatmaps.

**Mapper**:
A person who authors Beat Saber beatmaps.
_Avoid_: Using "mapper" for a trained model when the distinction matters.

### Gameplay objects

**Note**:
A colored gameplay object assigned to one saber, with a position and either a
required cut direction or unrestricted direction (a dot note).

**Bomb**:
A gameplay object that the player must avoid striking with either saber.

**Obstacle**:
A wall or other avoidance region occupying part of the playfield for a duration.

**Arc**:
A curved connection between gameplay endpoints for one saber. Existing code and
Beat Saber data use "slider" for this concept; use "arc" in prose to distinguish
it from a UI slider.

**Burst chain**:
A colored head followed by slices along one saber's swing. A burst chain is
distinct from an arc connecting existing notes and from staggered ordinary notes.

**Lighting event**:
A timed instruction controlling the level's lighting or related visual effects.
_Avoid_: Using "event" for a note when discussing the lighting pipeline.

### Timing and density

**Beat time**:
A position on the song's musical timeline expressed in beats. Conversion to
elapsed seconds depends on the tempo and any tempo changes.

**BPM**:
Beats per minute, a measure of musical tempo.

**Note density**:
The number of notes per second over a stated interval. A reported density must
make its interval and treatment of simultaneous notes explicit.

**BPS (beats per second)**:
Beats per second, used in InfernoSaber both as a beat-density measure and as an
approximate difficulty control. Specify "measured BPS" or "BPS difficulty setting"
when the distinction matters.

**Measured BPS**:
The beat density of a beatmap over a stated interval, expressed in beats per
second. State the interval and counting rule, including how simultaneous notes
and bombs are treated, when reporting or comparing measurements.

**BPS difficulty setting**:
The app's approximate difficulty control expressed in BPS; it does not guarantee
that a generated beatmap's measured BPS will match the setting. The expected
working range is approximately 1–10; values outside this range are supported but
are not expected to perform well.

### New beat-learning framework

**Retained note**:
The representative colored note kept for one saber in a synchronized event.
Original notes remain available for future expansion; bombs are excluded.

**Synchronized event**:
A group of nearby note onsets represented by one timestamp. The earliest onset
anchors the group; notes within the minimum cut interval are considered
simultaneous. A red/blue pair remains two notes within this one event.

**Requested NPS**:
The new framework's requested retained colored notes per second over the full
decoded song duration. A red/blue pair counts as two. This is distinct from the
legacy app's approximate BPS difficulty setting and from unique onset density.

**Intensity share**:
A snippet's fraction of the song's retained notes. Shares sum to one for a
nonempty target. Multiplying by requested NPS and song duration gives the
snippet's requested note budget; dividing by snippet duration gives local NPS.
BPM remains a separate tempo feature.

See [the framework contracts](docs/beat-model-training.md) for reduction, timing,
empty-target masks, and supported source formats.
