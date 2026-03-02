using System.Diagnostics.CodeAnalysis;
using Aiursoft.CppRunner.Entities;
using Microsoft.EntityFrameworkCore;

namespace Aiursoft.CppRunner.Sqlite;

[ExcludeFromCodeCoverage]

public class SqliteContext(DbContextOptions<SqliteContext> options) : TemplateDbContext(options)
{
    public override Task<bool> CanConnectAsync()
    {
        return Task.FromResult(true);
    }
}
