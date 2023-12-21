namespace Aiursoft.CppRunner.Lang;

public class JavaLang : ILang
{
    public string LangDisplayName { get; set; } = "Java (OpenJDK 23)";

    public string LangExtension { get; set; } = "java";

    public string LangName { get; set; } = "java";

    public string DefaultCode { get; set; } =
        """
        import java.util.stream.Stream;

        public class Main {
            public static void main(String[] args) {
                Stream.iterate(new int[]{1, 1}, i -> new int[]{i[1], i[0] + i[1]})
                        .map(i -> i[0])
                        .limit(20)
                        .forEach(System.out::println);
            }
        }
        """;

    public string EntryFileName { get; set; } = "Main.java";

    public string DockerImage { get; set; } = "openjdk:23-jdk";

    public string RunCommand { get; set; } = "javac /app/Main.java && java -cp /app Main";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}