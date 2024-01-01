# CppRunner

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://gitlab.aiursoft.cn/aiursoft/cpprunner/-/blob/master/LICENSE)
[![Pipeline stat](https://gitlab.aiursoft.cn/aiursoft/cpprunner/badges/master/pipeline.svg)](https://gitlab.aiursoft.cn/aiursoft/cpprunner/-/pipelines)
[![Test Coverage](https://gitlab.aiursoft.cn/aiursoft/cpprunner/badges/master/coverage.svg)](https://gitlab.aiursoft.cn/aiursoft/cpprunner/-/pipelines)
[![ManHours](https://manhours.aiursoft.cn/r/gitlab.aiursoft.cn/aiursoft/cpprunner.svg)](https://gitlab.aiursoft.cn/aiursoft/cpprunner/-/commits/master?ref_type=heads)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fcpprunner.aiursoft.cn%2F)](https://cpprunner.aiursoft.cn)

CppRunner is a simple Web API that can run C++ code for you. It's based on .NET and Docker.

## Try

Try a running CppRunner [here](https://cpprunner.aiursoft.cn).

## Run in Ubuntu

The following script will install\update this app on your Ubuntu server. Supports Ubuntu 22.04.

On your Ubuntu server, run the following command:

```bash
curl -sL https://gitlab.aiursoft.cn/aiursoft/cpprunner/-/raw/master/install.sh | sudo bash
```

Of course it is suggested that append a custom port number to the command:

```bash
curl -sL https://gitlab.aiursoft.cn/aiursoft/cpprunner/-/raw/master/install.sh | sudo bash -s 8080
```

It will install the app as a systemd service, and start it automatically. Binary files will be located at `/opt/apps`. Service files will be located at `/etc/systemd/system`.

## Run locally

Requirements about how to run

1. Install [Docker](https://www.docker.com/)
2. Install [.NET 7 SDK](http://dot.net/)
3. Install [Node.js](https://nodejs.org/)
4. Configure `www-data` user in your host machine to allow access to Docker with: `sudo usermod -aG docker www-data`
5. Add `www-data` home: `sudo mkdir /var/www && sudo chown www-data:www-data /var/www`
6. Run `npm install`   in `src/Aiursoft.CppRunner.FrontEnd` folder.
7. Run `npm run build` in `src/Aiursoft.CppRunner.FrontEnd` folder.
8. Copy `src/Aiursoft.CppRunner.FrontEnd/dist` folder to `src/Aiursoft.CppRunner/wwwroot` folder.
9. Execute `sudo -u www-data dotnet run` as www-data user in the project path.
10. Use your browser to view [http://localhost:5000](http://localhost:5000)

## Run in Microsoft Visual Studio

1. Open the `.sln` file in the project path.
2. Press `F5`.

## How to contribute

There are many ways to contribute to the project: logging bugs, submitting pull requests, reporting issues, and creating suggestions.

Even if you with push rights on the repository, you should create a personal fork and create feature branches there when you need them. This keeps the main repository clean and your workflow cruft out of sight.

We're also interested in your feedback on the future of this project. You can submit a suggestion or feature request through the issue tracker. To make this process more effective, we're asking that these include more information to help define them more clearly.