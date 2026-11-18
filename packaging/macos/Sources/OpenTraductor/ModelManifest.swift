import Foundation

/// Lista fija de lo que baja la descarga guiada. Las revisiones TIENEN que coincidir con runtime/config.py
/// (PARAKEET_REV, KOKORO_REV, MARIAN_REV). Tamaños y checksums salen de la API de Hugging Face
/// (`/api/models/<repo>/revision/<rev>?blobs=true`): sha256 para los archivos LFS y, para los chicos que no son LFS,
/// el hash git del blob (SHA-1 de "blob <tamaño>\0<contenido>"). Todo son datos (pesos y tokenizadores), no código.
enum Checksum {
    case sha256(String)
    case gitBlobSHA1(String)
}

struct ModelFile {
    let repo: String
    let revision: String
    /// Nombre que ve la persona en la pantalla de descarga.
    let label: String
    /// Ruta dentro del repositorio (y dentro del snapshot).
    let path: String
    let size: Int64
    let checksum: Checksum
    /// Archivos que no vienen de Hugging Face (Moonshine): URL completa y ruta relativa a Application Support/OpenTraductor.
    var directURL: URL? = nil
    var supportPath: String? = nil

    var url: URL { directURL ?? URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(path)")! }

    /// Misma estructura que la caché de Hugging Face (`<HF_HOME>/hub/models--org--repo/snapshots/<rev>/<archivo>`):
    /// el motor Python los encuentra sin cambios (runtime/config.py `snapshot()`).
    func destination(hub: URL) -> URL {
        if let supportPath { return RuntimeProcess.supportDir().appendingPathComponent(supportPath) }
        return hub.appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
            .appendingPathComponent(revision, isDirectory: true)
            .appendingPathComponent(path)
    }
}

enum ModelManifest {
    static let parakeetRepo = "mlx-community/parakeet-tdt-0.6b-v3"
    static let parakeetRev = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
    static let kokoroRepo = "mlx-community/Kokoro-82M-bf16"
    static let kokoroRev = "a71e4d38b236d968966a2002c4c895dbd12b1c3c"
    static let marianRepo = "Helsinki-NLP/opus-mt-es-en"
    static let marianRev = "c96e2c5399ebfae4fc43d9669556b9afa74bb69d"

    /// `<Application Support>/OpenTraductor/hf/hub` (HF_HOME = .../hf).
    static func hubDir() -> URL {
        RuntimeProcess.supportDir().appendingPathComponent("hf", isDirectory: true).appendingPathComponent("hub", isDirectory: true)
    }

    /// Traductor ya convertido (CTranslate2 int8, ~80 MB) dentro del bundle (`Resources/marian_ct2`, lo arma build_app.sh).
    /// Si está, no hace falta bajar ni convertir Marian: el motor lo lee de ahí con OPENTRADUCTOR_MARIAN_CT2.
    static func bundledMarianDir() -> URL? {
        let d = Bundle.main.resourceURL?.appendingPathComponent("marian_ct2", isDirectory: true)
        guard let d, FileManager.default.fileExists(atPath: d.appendingPathComponent("model.bin").path) else { return nil }
        return d
    }

    static func installedMarianDir() -> URL {
        RuntimeProcess.supportDir().appendingPathComponent("models/opus-mt-es-en-ct2-int8", isDirectory: true)
    }

    private static func lfs(_ repo: String, _ rev: String, _ label: String, _ path: String, _ size: Int64, _ sha: String) -> ModelFile {
        ModelFile(repo: repo, revision: rev, label: label, path: path, size: size, checksum: .sha256(sha))
    }

    private static func git(_ repo: String, _ rev: String, _ label: String, _ path: String, _ size: Int64, _ sha1: String) -> ModelFile {
        ModelFile(repo: repo, revision: rev, label: label, path: path, size: size, checksum: .gitBlobSHA1(sha1))
    }

    /// Reconocimiento de voz (Parakeet) y voz en inglés (Kokoro): lo mínimo que pide `runtime/preflight.py`.
    static let engineFiles: [ModelFile] = {
        let p = (parakeetRepo, parakeetRev, "Reconocimiento de voz")
        let k = (kokoroRepo, kokoroRev, "Voz en inglés")
        var out: [ModelFile] = [
            git(p.0, p.1, p.2, "config.json", 244_093, "4f469c2e92c981861f7ce6bcd940e608401d931e"),
            git(p.0, p.1, p.2, "tokenizer.vocab", 101_024, "3fa4c819f33b03e876ce33c0aa34866ed2b5e17a"),
            git(p.0, p.1, p.2, "vocab.txt", 46_772, "d2fc51742d86127c241018b728e72b3a336225a1"),
            lfs(p.0, p.1, p.2, "tokenizer.model", 360_916, "eacec2b0a77f336d4a2ca4a25a7047575d3c2b74de47e997f4c205126ed3135e"),
            git(k.0, k.1, k.2, "config.json", 2_351, "14a726edd3718279eac426630879ff743955b16a"),
        ]
        let voices: [(String, String)] = [
            ("af_heart", "2c1c733b0e6576c810e268d3e440c21dea4e0f0131a3ba4cfc98d7fe6136d094"),
            ("af_bella", "112d310468cbb3cf23404d3d0b50ad3adf017b87bf38bf9edd15f4ad572df6a3"),
            ("af_nicole", "574656386022c81a029e9a72558191925f44c3de2dad2fa2e45751938557d062"),
            ("am_michael", "3940147ded35deba0bb52e8132f89b719298e0520258c34584358aa5a24da2ea"),
            ("am_adam", "a4f60a3b9c20353c2604a17485ba53260502a758681a84d41e8af53cc559d929"),
        ]
        for (name, sha) in voices { out.append(lfs(k.0, k.1, k.2, "voices/\(name).safetensors", 522_320, sha)) }
        // Los pesos grandes van al final: si algo falla en los archivos chicos, se nota enseguida.
        out.append(lfs(k.0, k.1, k.2, "kokoro-v1_0.safetensors", 327_115_152, "4e9ecdf03b8b6cf906070390237feda473dc13327cb8d56a43deaa374c02acd8"))
        out.append(lfs(p.0, p.1, p.2, "model.safetensors", 2_508_288_736, "05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592"))
        return out
    }()

    /// Marian original (~315 MB). Solo se baja si el bundle no trae el traductor ya convertido.
    static let marianSourceFiles: [ModelFile] = {
        let m = (marianRepo, marianRev, "Traductor")
        return [
            git(m.0, m.1, m.2, "config.json", 1_438, "03e24f5988fe1a2b9bcf9f3bbebc9bc3b4e2ac1a"),
            git(m.0, m.1, m.2, "generation_config.json", 293, "f07859fcab98712b231d115475990a962eebc851"),
            git(m.0, m.1, m.2, "metadata.json", 1_090, "e0993f2fcafec0e6632b8fb9f38e2ecbb9884b41"),
            git(m.0, m.1, m.2, "source.spm", 825_924, "2dfb31564bc31c8fbc65c8bfa8cc201fbbf45e44"),
            git(m.0, m.1, m.2, "target.spm", 801_636, "6dc92ba66940f622779f82c376066c2c46bbad12"),
            git(m.0, m.1, m.2, "tokenizer_config.json", 44, "8325c8e127f51a74de059c5db69becaf36971327"),
            git(m.0, m.1, m.2, "vocab.json", 1_590_040, "d2c5a0f1ab2bb3a752894b854150c358f8c682f9"),
            lfs(m.0, m.1, m.2, "pytorch_model.bin", 312_087_523, "586a4e69e0804459ac7c176a5dc76852652100d9af1ee0db796de719899a5deb"),
        ]
    }()

    static func filesToDownload() -> [ModelFile] {
        let p = EngineProfile.current
        guard p.isStoreStack else {
            return bundledMarianDir() == nil ? engineFiles + marianSourceFiles : engineFiles
        }
        // Tienda (T-49): Moonshine + Qwen + Kokoro. Sin Parakeet (2,5 GB) ni Marian.
        let kokoro = engineFiles.filter { $0.repo == kokoroRepo }
        let small = { (f: ModelFile) in f.size < 50_000_000 }
        let asr = moonshineFiles[p.asrBackend] ?? []
        let mt = qwenFiles[p.qwenModel] ?? []
        let all = kokoro + asr + mt
        return all.filter(small) + all.filter { !small($0) }  // los pesos grandes al final
    }

    // MARK: Tienda (T-49): Moonshine + Qwen

    /// Revisión de Moonshine: TIENE que coincidir con runtime/config.py MOONSHINE_REV. sha256 de scripts/setup_moonshine.py.
    static let moonshineRev = "quantized_26_08_24"

    private static func moon(_ name: String, _ file: String, _ size: Int64, _ sha: String) -> ModelFile {
        let rel = "\(name)/\(moonshineRev)/\(file)"
        return ModelFile(repo: "moonshine", revision: moonshineRev, label: "Reconocimiento de voz", path: file, size: size,
                         checksum: .sha256(sha), directURL: URL(string: "https://download.moonshine.ai/model/\(rel)")!,
                         supportPath: "moonshine/\(rel)")  // OPENTRADUCTOR_MOONSHINE_DIR = <support>/moonshine
    }

    static let moonshineFiles: [String: [ModelFile]] = [
        "moonshine-tiny": [
            moon("tiny-streaming-es", "adapter.ort", 1_318_472, "2efd97cd2ee7d89578dd1d429a849be9fe37ac81dd9c4e1d31c8a2e7f85d43e6"),
            moon("tiny-streaming-es", "cross_kv.ort", 1_288_120, "9e268b1b3bb0f9ff79bca6e76055d41298de0a5ba52505ee04907a2642efa3ae"),
            moon("tiny-streaming-es", "decoder_kv.ort", 19_717_336, "7adf56b982df98e385d71823b9a40cf9296ae3a746efcf03f3d7bb4604032db0"),
            moon("tiny-streaming-es", "encoder.ort", 7_772_792, "ec9eb43a01fbfe242a2c6a86050cffe4a31050e1c9f084e98560d6c100004505"),
            moon("tiny-streaming-es", "frontend.model.ort", 23_176, "9d19434b861de0c1d2aff52556911ae565cf3212ebe6d7f797c76c5c5066c2fc"),
            moon("tiny-streaming-es", "frontend.weights.ort", 2_093_280, "5e4d5471cd5d984d4b94a603d861539ebf1dc72de11b0cf13f38316725d5ce48"),
            moon("tiny-streaming-es", "streaming_config.json", 509, "d31afd7c2dd5ee9ae3fa232312c5b42cb3e4fe1c6e9c53d885f2bea63023790a"),
            moon("tiny-streaming-es", "tokenizer.bin", 102_888, "5fbb7d4314dcb18e03c5f975609e4a4cd572b22b01d2b2accb1b3e6830696f36"),
        ],
        "moonshine-small": [
            moon("small-streaming-es", "adapter.ort", 2_869_296, "04b54114c8aab534222922640f7ca0882948ff9f6ed76777f6e184e55a8e8b15"),
            moon("small-streaming-es", "cross_kv.ort", 5_358_752, "4bfd0a641d72ccdae22751f86bbb2e25ff70c1fba4f81213f2415a1accff9618"),
            moon("small-streaming-es", "decoder_kv.ort", 61_314_512, "5b77c3d6baf801ef925a5bc54d7eb3db0c35ebbb86f8eaf5390d7f5bb42fef37"),
            moon("small-streaming-es", "encoder.ort", 44_358_376, "a9b8d6d5d9348d0e319cceffdb0196ef622df8e4e9f3fb4797dcd8dfafd50857"),
            moon("small-streaming-es", "frontend.model.ort", 26_776, "69c76287f49db365aa278d4908ec450e69ca1b4bfb7e836159d963d623ee0c13"),
            moon("small-streaming-es", "frontend.weights.ort", 7_769_280, "2f5a0eb5f3004c9447810d74274a352746319a32144e31d95ef410f425b84dda"),
            moon("small-streaming-es", "streaming_config.json", 512, "12d16c7f5ea6734d197b79baf47914ea7e996d6fca8d303a19aca10d0617cecc"),
            moon("small-streaming-es", "tokenizer.bin", 102_888, "5fbb7d4314dcb18e03c5f975609e4a4cd572b22b01d2b2accb1b3e6830696f36"),
        ],
    ]

    /// Qwen MLX por variante (T-50 elige). Revisiones fijas; tamaños y hashes de la API de HF (?blobs=true), 16/11.
    static let qwenRepos: [String: (repo: String, rev: String)] = [
        "qwen3-4b-instruct-2507-4bit": ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"),
        "qwen3-1.7b-4bit": ("mlx-community/Qwen3-1.7B-4bit", "3b1b1768f8f8cf8351c712464f906e86c2b8269e"),
        "qwen3-0.6b-4bit": ("mlx-community/Qwen3-0.6B-4bit", "73e3e38d981303bc594367cd910ea6eb48349da8"),
    ]

    /// Carpeta del snapshot de Qwen (lo que recibe el motor en OPENTRADUCTOR_QWEN_DIR).
    static func qwenDir(_ variant: String) -> URL? {
        guard let r = qwenRepos[variant] else { return nil }
        return hubDir().appendingPathComponent("models--" + r.repo.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true).appendingPathComponent(r.rev, isDirectory: true)
    }

    static let qwenFiles: [String: [ModelFile]] = {
        func files(_ v: String, _ list: [(String, Int64, Checksum)]) -> [ModelFile] {
            let r = qwenRepos[v]!
            return list.map { ModelFile(repo: r.repo, revision: r.rev, label: "Traductor", path: $0.0, size: $0.1, checksum: $0.2) }
        }
        // Comunes a las 3 variantes (mismo tokenizador Qwen3).
        let tok: [(String, Int64, Checksum)] = [
            ("added_tokens.json", 707, .gitBlobSHA1("b54f9135e44c1e81047e8d05cb027af8bc039eed")),
            ("special_tokens_map.json", 613, .gitBlobSHA1("ac23c0aaa2434523c494330aeb79c58395378103")),
            ("merges.txt", 1_671_853, .gitBlobSHA1("31349551d90c7606f325fe0f11bbb8bd5fa0d7c7")),
            ("vocab.json", 2_776_833, .gitBlobSHA1("4783fe10ac3adce15ac8f358ef5462739852c569")),
            ("tokenizer.json", 11_422_654, .sha256("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4")),
        ]
        return [
            "qwen3-4b-instruct-2507-4bit": files("qwen3-4b-instruct-2507-4bit", [
                ("config.json", 938, .gitBlobSHA1("ce8b8eccd1cdf6d8a30767f58e8ff858dd15eab5")),
                ("generation_config.json", 238, .gitBlobSHA1("432531a002c181a19de338313d2375e9d7494d7e")),
                ("chat_template.jinja", 4_040, .gitBlobSHA1("a18870ad4ba26ac6c43758fc506c1bb6ff206bb4")),
                ("tokenizer_config.json", 5_440, .gitBlobSHA1("474bbcd82077828bdec32b8dbc1826cdff2a792a")),
                ("model.safetensors.index.json", 63_964, .gitBlobSHA1("4741a210f9920c2949ca73bbb2a7ce9583e7fd83")),
            ] + tok + [
                ("model.safetensors", 2_263_022_417, .sha256("2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f")),
            ]),
            "qwen3-1.7b-4bit": files("qwen3-1.7b-4bit", [
                ("config.json", 937, .gitBlobSHA1("0a78ffc3980b062021a450199988d0ed8537239d")),
                ("tokenizer_config.json", 9_706, .gitBlobSHA1("7345216a0785dc7086e8c245b2a9d3896ce2b756")),
                ("model.safetensors.index.json", 49_731, .gitBlobSHA1("8607d041b6549c15a4db85e7b4c5cf30d3ab890a")),
            ] + tok + [
                ("model.safetensors", 968_080_210, .sha256("0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f")),
            ]),
            "qwen3-0.6b-4bit": files("qwen3-0.6b-4bit", [
                ("config.json", 937, .gitBlobSHA1("6c974b02334f2f5f68de4cb2796ee4fe80bfc712")),
                ("tokenizer_config.json", 9_706, .gitBlobSHA1("7345216a0785dc7086e8c245b2a9d3896ce2b756")),
                ("model.safetensors.index.json", 49_731, .gitBlobSHA1("e743a6a7d2ba4fd21b2256d739117cbf31345297")),
            ] + tok + [
                ("model.safetensors", 335_450_584, .sha256("392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2")),
            ]),
        ]
    }()
}

/// Qué motores usa este build. Sale del Info.plist (OTASRBackend, OTMTEngine, OTQwenModel); sin claves, el stack de
/// antes (Parakeet + Marian). Cambiar el ASR (T-48) o el tamaño de Qwen (T-50) es editar el Info.plist, no el código.
struct EngineProfile {
    let asrBackend: String
    let mtEngine: String
    let qwenModel: String

    var isStoreStack: Bool { mtEngine == "qwen" && asrBackend.hasPrefix("moonshine") }

    /// Argumentos del motor para `run` o `check` (mismo perfil en los dos).
    func runtimeArgs(_ cmd: String) -> [String] {
        ["-m", "runtime", cmd, "--asr-backend", asrBackend]
    }

    static var current: EngineProfile { from(Bundle.main.infoDictionary ?? [:]) }

    /// T-50: "auto" elige por RAM física, igual que runtime/config.py (QWEN_4B_MIN_RAM = 16 GiB): 4B con ≥16 GiB, si no 1.7B.
    static let qwenAuto = "auto"
    static let qwen4BMinRAM: UInt64 = 16 * 1024 * 1024 * 1024

    static func qwenModelForRAM(_ memsize: UInt64?) -> String {
        if let m = memsize, m >= qwen4BMinRAM { return "qwen3-4b-instruct-2507-4bit" }
        return "qwen3-1.7b-4bit"
    }

    static func macMemsize() -> UInt64? {
        var v: UInt64 = 0
        var n = MemoryLayout<UInt64>.size
        return sysctlbyname("hw.memsize", &v, &n, nil, 0) == 0 ? v : nil
    }

    static func from(_ info: [String: Any], memsize: UInt64? = macMemsize()) -> EngineProfile {
        let asr = info["OTASRBackend"] as? String ?? "parakeet"
        let mt = info["OTMTEngine"] as? String ?? "marian"
        let raw = info["OTQwenModel"] as? String ?? "qwen3-4b-instruct-2507-4bit"
        let q = raw == qwenAuto ? qwenModelForRAM(memsize) : raw
        let validASR = asr == "parakeet" || ModelManifest.moonshineFiles[asr] != nil
        let validQwen = ModelManifest.qwenRepos[q] != nil
        return EngineProfile(asrBackend: validASR ? asr : "parakeet", mtEngine: (mt == "qwen" && validQwen) ? "qwen" : "marian",
                             qwenModel: validQwen ? q : "qwen3-4b-instruct-2507-4bit")
    }
}
