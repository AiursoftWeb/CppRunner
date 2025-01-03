ARG CSPROJ_PATH="./src/Aiursoft.CppRunner/"
ARG PROJ_NAME="Aiursoft.CppRunner"
ARG FRONT_END_PATH="./src/Aiursoft.CppRunner.Frontend/"
# ============================
# Prepare NPM Environment
FROM hub.aiursoft.cn/node:21-alpine AS npm-env
ARG FRONT_END_PATH
WORKDIR /src
COPY . .

# NPM Build at PGK_JSON_PATH
RUN npm install --prefix "${FRONT_END_PATH}" --force --loglevel verbose
RUN npm run build --prefix "${FRONT_END_PATH}"

# ============================
# Prepare Building Environment
FROM hub.aiursoft.cn/mcr.microsoft.com/dotnet/sdk:9.0 AS build-env
ARG CSPROJ_PATH
ARG FRONT_END_PATH
ARG PROJ_NAME
WORKDIR /src
COPY --from=npm-env /src .

# Build
RUN dotnet publish ${CSPROJ_PATH}${PROJ_NAME}.csproj  --configuration Release --no-self-contained --runtime linux-x64 --output /app
RUN mkdir -p /app/wwwroot
RUN cp -r ${FRONT_END_PATH}/dist/* /app/wwwroot

# ============================
# Prepare Runtime Environment
FROM hub.aiursoft.cn/mcr.microsoft.com/dotnet/aspnet:9.0
ARG PROJ_NAME
WORKDIR /app
COPY --from=build-env /app .

# Install wget and curl
RUN apt update; DEBIAN_FRONTEND=noninteractive apt install -y wget curl

# Install Docker
RUN curl -fsSL https://get.docker.com -o get-docker.sh
ENV CHANNEL=stable
RUN sh get-docker.sh
RUN usermod -aG docker root
RUN rm get-docker.sh

# Edit appsettings.json
RUN sed -i 's/DataSource=app.db/DataSource=\/data\/app.db/g' appsettings.json
RUN sed -i 's/\/tmp\/data/\/data/g' appsettings.json
RUN mkdir -p /data

VOLUME /data
EXPOSE 5000

ENV SRC_SETTINGS=/app/appsettings.json
ENV VOL_SETTINGS=/data/appsettings.json
ENV DLL_NAME=${PROJ_NAME}.dll

#ENTRYPOINT dotnet $DLL_NAME --urls http://*:5000
ENTRYPOINT ["/bin/bash", "-c", "\
    if [ ! -f \"$VOL_SETTINGS\" ]; then \
        cp $SRC_SETTINGS $VOL_SETTINGS; \
    fi && \
    if [ -f \"$SRC_SETTINGS\" ]; then \
        rm $SRC_SETTINGS; \
    fi && \
    ln -s $VOL_SETTINGS $SRC_SETTINGS && \
    dotnet $DLL_NAME --urls http://*:5000 \
"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=180s --retries=3 CMD \
wget --quiet --tries=1 --spider http://localhost:5000/health || exit 1