using System.Reflection;
using Aiursoft.Canon;
using Aiursoft.CppRunner.Lang;
using Aiursoft.CppRunner.Services;
using Aiursoft.CSTools.Services;
using Aiursoft.WebTools.Abstractions.Models;

namespace Aiursoft.CppRunner;

public class Startup : IWebStartup
{
    public virtual void ConfigureServices(IConfiguration configuration, IWebHostEnvironment environment, IServiceCollection services)
    {
        services.AddScoped<CommandService>();
        services.AddScoped<RunCodeService>();

        services.AddTaskCanon();
        services.AddScoped<ILang, CLang>();
        services.AddScoped<ILang, CppLang>();
        services.AddScoped<ILang, CudaLang>();
        services.AddScoped<ILang, CSharpLang>();
        
        services.AddScoped<ILang, GoLang>();
        services.AddScoped<ILang, RustLang>();
        
        services.AddScoped<ILang, JavaScriptLang>();
        services.AddScoped<ILang, TypeScriptLang>();
        
        services.AddScoped<ILang, PythonLang>();
        services.AddScoped<ILang, PythonWithPytorch>();
        services.AddScoped<ILang, BashLang>();
        services.AddScoped<ILang, PowerShellLang>();
        
        services.AddScoped<ILang, SwiftLang>();
        
        services.AddScoped<ILang, JavaLang>();
        services.AddScoped<ILang, RubyLang>();
        
        services.AddScoped<ILang, PhpLang>();
        
        services.AddScoped<ILang, PerlLang>();
        services.AddScoped<ILang, LuaLang>();
        
        services.AddScoped<ILang, HaskellLang>();
        services.AddScoped<ILang, LispLang>();


        services
            .AddControllersWithViews()
            .AddApplicationPart(Assembly.GetExecutingAssembly());
    }

    public void Configure(WebApplication app)
    {
        app.UseDefaultFiles();
        app.UseStaticFiles();
        app.UseRouting();
        app.MapDefaultControllerRoute();
    }
}