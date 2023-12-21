namespace Aiursoft.CppRunner.Lang;

public class LuaLang : ILang
{
    public string LangDisplayName { get; set; } = "Lua (Lua 5.4)";

    public string LangExtension { get; set; } = "lua";

    public string LangName { get; set; } = "lua";

    public string DefaultCode { get; set; } =
        """
        function fibonacci()
            local current, next = 1, 1
            return function()
                local result = current
                current, next = next, current + next
                return result
            end
        end

        local fib = fibonacci()
        for i = 1, 20 do
            print(fib())
        end
        """;

    public string EntryFileName { get; set; } = "main.lua";

    public string DockerImage { get; set; } = "lua:5.4";

    public string RunCommand { get; set; } = "lua /app/main.lua";

    public Dictionary<string, string> OtherFiles { get; set; } = new();
}