using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace Aiursoft.CppRunner.Sqlite.Migrations
{
    /// <inheritdoc />
    public partial class AddResultAndErrorToSavedCode : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "Error",
                table: "SavedCodes",
                type: "TEXT",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "Result",
                table: "SavedCodes",
                type: "TEXT",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Error",
                table: "SavedCodes");

            migrationBuilder.DropColumn(
                name: "Result",
                table: "SavedCodes");
        }
    }
}
