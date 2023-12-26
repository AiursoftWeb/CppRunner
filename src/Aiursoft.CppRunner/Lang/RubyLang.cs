namespace Aiursoft.CppRunner.Lang;

public class RubyLang : ILang
{
    public string LangDisplayName => "Ruby (3.2.2)";

    public string LangExtension => "rb";

    public string LangName => "ruby";

    public string DefaultCode =>
        """
        def fibonacci(n)
          return n if n <= 1
          fibonacci(n - 1) + fibonacci(n - 2)
        end

        20.times do |n|
          result = fibonacci(n)
          puts result
        end

        """;

    public string EntryFileName => "main.rb";

    public string DockerImage => "ruby:3.2.2";

    public string RunCommand => "ruby /app/main.rb";

    public Dictionary<string, string> OtherFiles => new();
}