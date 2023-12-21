using Aiursoft.WebTools;

namespace Aiursoft.CppRunner;

public class Program
{
    public static async Task Main(string[] args)
    {
        var app = Extends.App<Startup>(args);
        await app.PullContainersAsync();
        await app.RunAsync();
    }
}