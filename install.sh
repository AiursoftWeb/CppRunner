aiur() { arg="$( cut -d ' ' -f 2- <<< "$@" )" && curl -sL https://gitlab.aiursoft.cn/aiursoft/aiurscript/-/raw/master/$1.sh | sudo bash -s $arg; }

app_name="cpprunner"
repo_path="https://gitlab.aiursoft.cn/aiursoft/cpprunner"
proj_path="src/Aiursoft.CppRunner/Aiursoft.CppRunner.csproj"

get_dll_name()
{
    filename=$(basename -- "$proj_path")
    project_name="${filename/.csproj/}"
    dll_name="$project_name.dll"
    echo $dll_name
}

install()
{
    port=$1
    if [ -z "$port" ]; then
        port=$(aiur network/get_port)
    fi
    echo "Installing $app_name... to port $port"

    # Install prerequisites    
    aiur install/docker
    aiur install/dotnet
    aiur install/node

    # Clone the repo
    aiur git/clone_to $repo_path /tmp/repo

    # Install node modules
    wwwrootPath=$(dirname "/tmp/repo/$proj_path")/wwwroot
    if [ -d "$wwwrootPath" ]; then
        echo "Found wwwroot folder $wwwrootPath, will install node modules."
        sudo npm install --prefix "$wwwrootPath" -force
    fi

    # Publish the app
    aiur dotnet/publish "/tmp/repo/$proj_path" "/opt/apps/$app_name"
    
    # Register the service
    usermod -aG docker www-data
    dll_name=$(get_dll_name)
    aiur services/register_aspnet_service $app_name $port "/opt/apps/$app_name" $dll_name

    # Clean up
    echo "Install $app_name finished! Please open http://$(hostname):$port to try!"
    sudo rm /tmp/repo -rf
}

# This will install this app under /opt/apps and register a new service with systemd.
# Example: install 5000
install "$@"