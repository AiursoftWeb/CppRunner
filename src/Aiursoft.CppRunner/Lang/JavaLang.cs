namespace Aiursoft.CppRunner.Lang;

public class JavaLang : ILang
{
    public string LangDisplayName => "Java (OpenJDK 23)";

    public string LangExtension => "java";

    public string LangName => "java";

    public string DefaultCode =>
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

    public string EntryFileName => "Main.java";

    public string DockerImage => "eclipse-temurin:24-jdk";

    public string RunCommand => "javac /app/Main.java && java -cp /app Main";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}
