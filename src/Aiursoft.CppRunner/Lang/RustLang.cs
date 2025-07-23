namespace Aiursoft.CppRunner.Lang;

public class RustLang : ILang
{
    public string LangDisplayName => "Rust (1.74.1)";

    public string LangExtension => "rust";

    public string LangName => "rust";

    public string DefaultCode =>
        """
        fn fibonacci() -> impl Iterator<Item = u64> {
            let mut current = 1;
            let mut next = 1;
            std::iter::from_fn(move || {
                let result = current;
                current = next;
                next = current + result;
                Some(result)
            })
        }

        fn main() {
            for i in fibonacci().take(20) {
                println!("{}", i);
            }
        }
        """;

    public string EntryFileName => "main.rs";

    public string DockerImage => "rust:1.74.1";

    public string RunCommand => "cd /app && rustc /app/main.rs && /app/main";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}
