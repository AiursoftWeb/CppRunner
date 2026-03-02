using System.Diagnostics.CodeAnalysis;
using Aiursoft.CppRunner.Entities;
using Microsoft.EntityFrameworkCore;

namespace Aiursoft.CppRunner.MySql;

[ExcludeFromCodeCoverage]

public class MySqlContext(DbContextOptions<MySqlContext> options) : TemplateDbContext(options);
