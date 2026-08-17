# TORGO Current Manual Audit

Module 0 must revalidate these values programmatically.

## Speakers
Dysarthric: F01, F03, F04, M01, M02, M03, M04, M05

Control: FC01, FC02, FC03, MC01, MC02, MC03, MC04

Total observed speakers: **15**

## Files
- total extracted: **82,772**
- WAV: **17,635**
- prompt records checked: **9,494**

## Microphone coverage
- both: **7,764**
- head only: **450**
- array only: **1,276**
- neither: **4**
- direct head WAVs: **8,218**
- direct array WAVs: **9,417**

Canonical sample = speaker/session/prompt. Audit both microphones; prefer structurally valid head audio, otherwise use structurally valid array audio with explicit fallback evidence.

Do not interpret clinical assessment letters as severity classes without verified documentation.
