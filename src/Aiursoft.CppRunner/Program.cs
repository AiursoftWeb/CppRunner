using System.Reflection;
using Aiursoft.Canon;
using Aiursoft.CppRunner.Lang;
using Aiursoft.CSTools.Services;
using Aiursoft.WebTools;
using Aiursoft.WebTools.Models;

namespace Aiursoft.CppRunner;

public class Program
{
    public static async Task Main(string[] args)
    {
        var app = Extends.App<Startup>(args);
        await app.RunAsync();
    }
}

public class Startup : IWebStartup
{
    public void ConfigureServices(IConfiguration configuration, IWebHostEnvironment environment, IServiceCollection services)
    {
        services.AddScoped<CommandService>();

        services.AddTaskCanon();
        
        services.AddScoped<ILang, CppLang>();
        services.AddScoped<ILang, CSharpLang>();
        
        services
            .AddControllersWithViews()
            .AddApplicationPart(Assembly.GetExecutingAssembly());
    }

    public void Configure(WebApplication app)
    {
        app.UseMiddleware<AllowCrossOriginMiddleware>();
        app.UseRouting();
        app.MapDefaultControllerRoute();
    }
}