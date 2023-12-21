namespace Aiursoft.CppRunner.Lang;

public interface ILang
{
    string LangName { get; set; }
    
    string LangExtension { get; set; }

    string DefaultCode { get; set; }

    string FileName { get; set; }

    string DockerImage { get; set; }

    string RunCommand { get; set; }
}

public class CppLang : ILang
{
    public string LangName { get; set; } = "C++";
    
    public string LangExtension { get; set; } = "cpp";

    public string DefaultCode { get; set; } = @"#include <iostream>
#include <functional>
std::function<int()> fibonacci()
{
    int current = 1, next = 1;
    return [=]() mutable {
        int result = current;
        current = next;
        next = current + result;
        return result;
    };
}
int main()
{
    std::cout << ""Hello world!"" << std::endl;

    auto fib = fibonacci();
    for (int i = 0; i < 10; i++) {
        std::cout << fib() << std::endl;
    }
    return 0;
}";

    public string FileName { get; set; } = "main.cpp";
    public string DockerImage { get; set; } = "frolvlad/alpine-gxx";
    public string RunCommand { get; set; } = "g++ /app/main.cpp -o /tmp/main && /tmp/main";
}