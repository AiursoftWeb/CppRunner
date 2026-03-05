using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Aiursoft.CppRunner.Sqlite.Migrations
{
    /// <inheritdoc />
    public partial class AddResultCodeToSavedCode : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<int>(
                name: "ResultCode",
                table: "SavedCodes",
                type: "INTEGER",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ResultCode",
                table: "SavedCodes");
        }
    }
}
