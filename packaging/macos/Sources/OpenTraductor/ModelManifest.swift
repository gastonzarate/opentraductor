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

    var url: URL { URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(path)")! }

    /// Misma estructura que la caché de Hugging Face (`<HF_HOME>/hub/models--org--repo/snapshots/<rev>/<archivo>`):
    /// el motor Python los encuentra sin cambios (runtime/config.py `snapshot()`).
    func destination(hub: URL) -> URL {
        hub.appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
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
        bundledMarianDir() == nil ? engineFiles + marianSourceFiles : engineFiles
    }
}
