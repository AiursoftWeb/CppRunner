namespace Aiursoft.CppRunner.Lang;

public class KotlinLang : ILang
{
    public string LangDisplayName { get; set; } = "Kotlin (mingc/android-build-box)";

    public string LangExtension { get; set; } = "kt";

    public string LangName { get; set; } = "kotlin";

    public string DefaultCode { get; set; } =
        """
        fun fibonacci(): Sequence<Int> {
            var current = 1
            var next = 1
            return generateSequence {
                val result = current
                current = next
                next = current + result
                result
            }
        }

        fun main() {
            fibonacci().take(20).forEach(::println)
        }
        """;

    public string EntryFileName { get; set; } = "Main.kt";

    public string DockerImage { get; set; } = "mingc/android-build-box:1.26.0";

    public string RunCommand { get; set; } =
        "kotlinc /app/Main.kt -include-runtime -d /tmp/main.jar && java -jar /tmp/main.jar";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}