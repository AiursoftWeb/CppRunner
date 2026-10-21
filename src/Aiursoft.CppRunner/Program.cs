using System.Diagnostics.CodeAnalysis;
using Aiursoft.Canon.TaskQueue;
using Aiursoft.CppRunner.Services.BackgroundJobs;
using Aiursoft.ClickhouseLoggerProvider;
using Aiursoft.DbTools;
using Aiursoft.CppRunner.Entities;
using static Aiursoft.WebTools.Extends;

namespace Aiursoft.CppRunner;

[ExcludeFromCodeCoverage]
public abstract class Program
{
    public static async Task Main(string[] args)
    {
        var app = await AppAsync<Startup>(args);
        await app.Services.InitLoggingTableAsync();
        await app.UpdateDbAsync<TemplateDbContext>();
        await app.SeedAsync();
        await app.CopyAvatarFileAsync();
        app.Services.GetRequiredService<ServiceTaskQueue>()
            .QueueWithDependency<RuntimeImageWarmupJob>(job => job.ExecuteAsync());
        await app.RunAsync();
    }
}
