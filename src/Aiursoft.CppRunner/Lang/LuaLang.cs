namespace Aiursoft.CppRunner.Lang;

public class LuaLang : ILang
{
    public string LangDisplayName => "Lua (5.4)";

    public string LangExtension => "lua";

    public string LangName => "lua";

    public string DefaultCode =>
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

    public string EntryFileName => "main.lua";

    public string DockerImage => "hub.aiursoft.cn/imolein/lua:5.4";

    public string RunCommand => "lua /app/main.lua";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}