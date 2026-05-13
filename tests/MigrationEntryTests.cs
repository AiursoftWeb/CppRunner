using Aiursoft.CppRunner.Models.SystemViewModels;

namespace Aiursoft.CppRunner.Tests;

[TestClass]
public class MigrationEntryTests
{
    [TestMethod]
    public void Name_ParsedCorrectly_FromStandardId()
    {
        var entry = new MigrationEntry { Id = "20260108110700_AddGlobalSettings" };
        Assert.AreEqual("AddGlobalSettings", entry.Name);
    }

    [TestMethod]
    public void Name_ReturnsFullId_WhenShorterThan15Chars()
    {
        var entry = new MigrationEntry { Id = "ShortId" };
        Assert.AreEqual("ShortId", entry.Name);
    }

    [TestMethod]
    public void AppliedAt_ParsedCorrectly_FromStandardId()
    {
        var entry = new MigrationEntry { Id = "20260108110700_AddGlobalSettings" };
        var expected = new DateTime(2026, 1, 8, 11, 7, 0, DateTimeKind.Utc);
        Assert.AreEqual(expected, entry.AppliedAt);
    }

    [TestMethod]
    public void AppliedAt_ReturnsNull_WhenTimestampIsInvalid()
    {
        var entry = new MigrationEntry { Id = "NotATimestamp_SomeMigration" };
        Assert.IsNull(entry.AppliedAt);
    }
}
