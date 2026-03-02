using System.ComponentModel.DataAnnotations;
using Aiursoft.CppRunner.Entities;
using Aiursoft.UiStack.Layout;

namespace Aiursoft.CppRunner.Models.UsersViewModels;

public class DeleteViewModel : UiStackLayoutViewModel
{
    public DeleteViewModel()
    {
        PageTitle = "Delete User";
    }

    [Display(Name = "User")]
    public required User User { get; set; }
}
