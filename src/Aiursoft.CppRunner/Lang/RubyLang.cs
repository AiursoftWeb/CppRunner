namespace Aiursoft.CppRunner.Lang;

public class RubyLang : ILang
{
    public string LangDisplayName { get; set; } = "Ruby (3.2.2)";

    public string LangExtension { get; set; } = "rb";

    public string LangName { get; set; } = "ruby";

    public string DefaultCode { get; set; } =
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

    public string EntryFileName { get; set; } = "main.rb";

    public string DockerImage { get; set; } = "ruby:3.2.2";

    public string RunCommand { get; set; } = "ruby /app/main.rb";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}