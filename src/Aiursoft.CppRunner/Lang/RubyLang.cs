namespace Aiursoft.CppRunner.Lang;

public class RubyLang : ILang
{
    public string LangDisplayName { get; set; } = "Ruby (3.2.2)";

    public string LangExtension { get; set; } = "rb";

    public string LangName { get; set; } = "ruby";

    public string DefaultCode { get; set; } =
        """
        def fibonacci
          Enumerator.new do |y|
            current, next = 1, 1
            loop do
              y << current
              current, next = next, current + next
            end
          end
        end
        """;

    public string EntryFileName { get; set; } = "main.rb";

    public string DockerImage { get; set; } = "ruby:3.2.2";

    public string RunCommand { get; set; } = "ruby /app/main.rb";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}