// 既定マイクを 16kHz mono int16 で取り込み、生 PCM を標準出力へ流す。livediar の MicSource が subprocess として使う。
import AVFoundation
import Foundation

setvbuf(stdout, nil, _IOFBF, 1 << 16)
let status = AVCaptureDevice.authorizationStatus(for: .audio)
if status == .notDetermined {
    let sem = DispatchSemaphore(value: 0)
    AVCaptureDevice.requestAccess(for: .audio) { _ in sem.signal() }
    sem.wait()
}
if AVCaptureDevice.authorizationStatus(for: .audio) != .authorized {
    FileHandle.standardError.write("micpipe: microphone permission denied for this terminal app\n".data(using: .utf8)!)
    exit(2)
}
let engine = AVAudioEngine()
let input = engine.inputNode
let fmt = input.inputFormat(forBus: 0)
guard fmt.sampleRate > 0 else { FileHandle.standardError.write("micpipe: no input device\n".data(using: .utf8)!); exit(3) }
let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)!
let converter = AVAudioConverter(from: fmt, to: target)!
let out = FileHandle.standardOutput
FileHandle.standardError.write("micpipe: capturing \(Int(fmt.sampleRate)) Hz -> 16000 Hz mono int16\n".data(using: .utf8)!)
input.installTap(onBus: 0, bufferSize: 1600, format: fmt) { buf, _ in
    let cap = AVAudioFrameCount(Double(buf.frameLength) * 16000 / fmt.sampleRate) + 16
    let pcm = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: cap)!
    var consumed = false
    var err: NSError?
    converter.convert(to: pcm, error: &err) { _, st in
        if consumed { st.pointee = .noDataNow; return nil }
        consumed = true; st.pointee = .haveData; return buf
    }
    let bytes = Int(pcm.frameLength) * 2
    if bytes > 0, let p = pcm.int16ChannelData?[0] {
        out.write(Data(bytes: p, count: bytes))
    }
}
try! engine.start()
signal(SIGINT) { _ in exit(0) }
signal(SIGTERM) { _ in exit(0) }
RunLoop.main.run()
