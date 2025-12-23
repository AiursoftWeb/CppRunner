namespace Aiursoft.CppRunner.Lang;

public class CppLang : ILang
{
    public string LangDisplayName => "C++ (GNU G++, stdc++20)";

    public string LangExtension => "cpp";

    public string LangName => "cpp";

    public string DefaultCode =>
        """
        #include <iostream>
        #include <functional>

        template <typename Sig>
        std::function<Sig> Fix(std::function<std::function<Sig>(std::function<Sig>)> f) {
            auto g = [f](auto x) -> std::function<Sig> {
                return [f, x](auto... args) {
                    return f(x(x))(args...);
                };
            };
            return g(g);
        }

        int main() {
            auto fibLogic = [](auto self) {
                return [self](int n) -> int {
                    return (n <= 1) ? n : self(n - 1) + self(n - 2);
                };
            };

            auto fib = Fix<int(int)>(fibLogic);

            std::cout << "fib(10) = " << fib(10) << std::endl;

            return 0;
        }
        """;

    public string EntryFileName => "main.cpp";
    public string DockerImage => "frolvlad/alpine-gxx:latest";
    public string RunCommand => "g++ -Wall -Wextra -O2 -std=c++20 /app/main.cpp -o /tmp/main && /tmp/main";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}
