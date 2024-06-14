namespace Aiursoft.CppRunner.Lang;

public class RubyLang : ILang
{
    public string LangDisplayName => "Ruby (3.2.2)";

    public string LangExtension => "ruby";

    public string LangName => "ruby";

    public string DefaultCode =>
        """
        def fibonacci(n)
          fib = [0, 1]
          (2..n).each do |i|
            fib[i] = fib[i - 1] + fib[i - 2]
          end
          fib[n]
        end

        20.times do |n|
          result = fibonacci(n)
          puts result
        end
        """;

    public string EntryFileName => "main.rb";

    public string DockerImage => "hub.aiursoft.cn/ruby:3.2.2";

    public string RunCommand => "ruby /app/main.rb";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}