using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Diagnostics.CodeAnalysis;

namespace Aiursoft.CppRunner.Entities;

[ExcludeFromCodeCoverage]
public class SavedCode
{
    [Key]
    public int Id { get; set; }

    [Required]
    public required string UserId { get; set; }

    [ForeignKey(nameof(UserId))]
    public required User User { get; set; }

    [Required]
    [MaxLength(255)]
    public required string Title { get; set; }

    [Required]
    public required string Code { get; set; }

    [Required]
    [MaxLength(50)]
    public required string Language { get; set; }

    public bool IsPublic { get; set; }

    public DateTime CreationTime { get; set; } = DateTime.UtcNow;
}
