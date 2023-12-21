using System.Reflection;
using Aiursoft.Canon;
using Aiursoft.CppRunner.Lang;
using Aiursoft.CSTools.Services;
using Aiursoft.WebTools.Models;

namespace Aiursoft.CppRunner;

public class Startup : IWebStartup
{
    public void ConfigureServices(IConfiguration configuration, IWebHostEnvironment environment, IServiceCollection services)
    {
        services.AddScoped<CommandService>();

        services.AddTaskCanon();
        services.AddScoped<ILang, CppLang>();
        services.AddScoped<ILang, CSharpLang>();
        services.AddScoped<ILang, PythonLang>();
        services.AddScoped<ILang, NodeJsLang>();
        
        services
            .AddControllersWithViews()
            .AddApplicationPart(Assembly.GetExecutingAssembly());
    }

    public void Configure(WebApplication app)
    {
        app.UseDefaultFiles();
        app.UseStaticFiles();
        app.UseRouting();
        app.UseMiddleware<AllowCrossOriginMiddleware>();
        app.MapDefaultControllerRoute();
    }
}