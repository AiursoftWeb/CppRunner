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
        services.AddScoped<ILang, TypeScriptLang>();
        services.AddScoped<ILang, GoLang>();
        services.AddScoped<ILang, JavaLang>();
        services.AddScoped<ILang, RustLang>();
        services.AddScoped<ILang, RubyLang>();
        services.AddScoped<ILang, PhpLang>();
        services.AddScoped<ILang, PerlLang>();
        services.AddScoped<ILang, SwiftLang>();
        services.AddScoped<ILang, KotlinLang>();
        services.AddScoped<ILang, HaskellLang>();
        // TODO: 
        // C
        // Lisp
        // Scala
        // Haskell
        
        
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