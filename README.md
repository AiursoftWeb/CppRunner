# CppRunner

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://gitlab.aiursoft.cn/aiursoft/cpprunner/-/blob/master/LICENSE)
[![Pipeline stat](https://gitlab.aiursoft.cn/aiursoft/cpprunner/badges/master/pipeline.svg)](https://gitlab.aiursoft.cn/aiursoft/cpprunner/-/pipelines)
[![Test Coverage](https://gitlab.aiursoft.cn/aiursoft/cpprunner/badges/master/coverage.svg)](https://gitlab.aiursoft.cn/aiursoft/cpprunner/-/pipelines)
[![ManHours](https://manhours.aiursoft.cn/r/gitlab.aiursoft.cn/aiursoft/cpprunner.svg)](https://gitlab.aiursoft.cn/aiursoft/cpprunner/-/commits/master?ref_type=heads)

CppRunner is a simple Web API that can run C++ code for you. It's based on .NET and Docker.

## Run locally

Requirements about how to run

1. Install [Docker](https://www.docker.com/)
2. Configure `www-data` user in your host machine to allow access to Docker with: `sudo usermod -aG docker www-data`
3. Install [.NET 7 SDK](http://dot.net/)
4. Execute `dotnet run` as www-data user in the project path.
5. Use your browser to view [http://localhost:5000](http://localhost:5000)

## Run in Microsoft Visual Studio

1. Open the `.sln` file in the project path.
2. Press `F5`.

## How to contribute

There are many ways to contribute to the project: logging bugs, submitting pull requests, reporting issues, and creating suggestions.

Even if you with push rights on the repository, you should create a personal fork and create feature branches there when you need them. This keeps the main repository clean and your workflow cruft out of sight.

We're also interested in your feedback on the future of this project. You can submit a suggestion or feature request through the issue tracker. To make this process more effective, we're asking that these include more information to help define them more clearly.