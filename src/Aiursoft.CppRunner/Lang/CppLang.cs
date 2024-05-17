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
            auto fib = fibonacci();
            for (int i = 0; i < 20; i++) {
                std::cout << fib() << std::endl;
            }
            return 0;
        }

        """;

    public string EntryFileName => "main.cpp";
    public string DockerImage => "hub.aiursoft.cn/frolvlad/alpine-gxx:latest";
    public string RunCommand => "g++ -Wall -Wextra -O2 -std=c++20 /app/main.cpp -o /tmp/main && /tmp/main";

    public Dictionary<string, string> OtherFiles => new();
    public bool NeedGpu => false;

}