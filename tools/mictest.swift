// マイクの動作確認。既定入力から N 秒録音して音量を表示し、16kHz mono WAV に保存する。
// swiftc -O -o tools/mictest tools/mictest.swift && ./tools/mictest [秒数=3] [出力=mic_test.wav]
import AVFoundation
import Foundation

let seconds = Double(CommandLine.arguments.dropFirst().first ?? "3") ?? 3
let outPath = CommandLine.arguments.dropFirst(2).first ?? "mic_test.wav"

// 1) 権限の状態を表示し、未確認ならダイアログを出す
let status = AVCaptureDevice.authorizationStatus(for: .audio)
let names = [0: "notDetermined(未確認)", 1: "restricted(制限)", 2: "denied(拒否)", 3: "authorized(許可)"]
print("mic permission: \(names[status.rawValue] ?? "\(status.rawValue)")")
if status == .notDetermined {
    let sem = DispatchSemaphore(value: 0)
    AVCaptureDevice.requestAccess(for: .audio) { ok in print("permission dialog result: \(ok ? "許可" : "拒否")"); sem.signal() }
    sem.wait()
} else if status == .denied || status == .restricted {
    print("このターミナルアプリはマイクが拒否されています。システム設定 > プライバシーとセキュリティ > マイク で許可してください。")
    print("または: tccutil reset Microphone <bundle id> で再度ダイアログを出せます")
    exit(2)
}

// 2) 録音
let engine = AVAudioEngine()
let input = engine.inputNode
let fmt = input.inputFormat(forBus: 0)
print("input device format: \(Int(fmt.sampleRate)) Hz, \(fmt.channelCount) ch")
guard fmt.sampleRate > 0 else { print("入力デバイスが取得できません (sampleRate=0)"); exit(3) }

let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)!
let converter = AVAudioConverter(from: fmt, to: target)!
let file = try! AVAudioFile(forWriting: URL(fileURLWithPath: outPath), settings: target.settings, commonFormat: .pcmFormatInt16, interleaved: true)

var peak: Float = 0, sumSq: Double = 0, count: Int = 0
let lock = NSLock()
input.installTap(onBus: 0, bufferSize: 4096, format: fmt) { buf, _ in
    if let ch = buf.floatChannelData?[0] {
        for i in 0..<Int(buf.frameLength) { let v = abs(ch[i]); if v > peak { peak = v }; sumSq += Double(v*v); count += 1 }
    }
    let cap = AVAudioFrameCount(Double(buf.frameLength) * 16000 / fmt.sampleRate) + 16
    let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: cap)!
    var consumed = false
    var err: NSError?
    converter.convert(to: out, error: &err) { _, st in
        if consumed { st.pointee = .noDataNow; return nil }
        consumed = true; st.pointee = .haveData; return buf
    }
    lock.lock(); try? file.write(from: out); lock.unlock()
}
try! engine.start()
print("recording \(Int(seconds)) s ... 何か話してください")
Thread.sleep(forTimeInterval: seconds)
engine.stop(); input.removeTap(onBus: 0)

let rms = count > 0 ? sqrt(sumSq / Double(count)) : 0
let dbfs = { (x: Double) -> String in x > 0 ? String(format: "%.1f dBFS", 20*log10(x)) : "-inf" }
print(String(format: "peak: %@   rms: %@   samples: %d", dbfs(Double(peak)), dbfs(rms), count))
if peak < 0.001 { print("→ ほぼ無音です。マイク権限が無い / 入力デバイスが違う / ミュート の可能性") }
else if peak < 0.02 { print("→ 音量がかなり小さいです。マイクに近づくか入力レベルを上げてください") }
else { print("→ 音声は届いています。保存: \(outPath)") }
