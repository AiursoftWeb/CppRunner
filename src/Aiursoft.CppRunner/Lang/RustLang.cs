namespace Aiursoft.CppRunner.Lang;

public class RustLang : ILang
{
    public string LangDisplayName { get; set; } = "Rust (1.74.1)";

    public string LangExtension { get; set; } = "rs";

    public string LangName { get; set; } = "rust";

    public string DefaultCode { get; set; } =
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

    public string EntryFileName { get; set; } = "main.rs";

    public string DockerImage { get; set; } = "rust:1.74.1";

    public string RunCommand { get; set; } = "cd /app && rustc /app/main.rs && /app/main";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}