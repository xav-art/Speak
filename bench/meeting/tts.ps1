# Makes the meeting benchmark clips with the Windows TTS voices (Mark, Zira,
# David, Helle). From the Speak folder: .\bench\meeting\tts.ps1 bench\meeting\conv.json bench\meeting\conv
# (and enroll.json -> enroll). One WAV per line, numbered in order.
param([string]$JsonPath, [string]$OutDir)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$t) { $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(-1) | Out-Null; $task.Result }
[Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.Streams.DataReader,Windows.Storage.Streams,ContentType=WindowsRuntime] | Out-Null
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$items = Get-Content -Raw -Encoding UTF8 $JsonPath | ConvertFrom-Json
$i = 0
foreach ($it in $items) {
  $synth.Voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.DisplayName -like "*$($it.voice)*" } | Select-Object -First 1
  $stream = Await ($synth.SynthesizeTextToStreamAsync($it.text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
  $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
  $n = Await ($reader.LoadAsync([uint32]$stream.Size)) ([uint32])
  $bytes = New-Object byte[] $n; $reader.ReadBytes($bytes)
  [IO.File]::WriteAllBytes((Join-Path $OutDir ("{0:D3}.wav" -f $i)), $bytes)
  $i++
}
"wrote $i files"
