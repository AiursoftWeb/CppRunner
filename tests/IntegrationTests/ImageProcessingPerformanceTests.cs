using System.Diagnostics;
using Aiursoft.CppRunner.Services.FileStorage;
using SkiaSharp;

namespace Aiursoft.CppRunner.Tests.IntegrationTests;

#pragma warning disable MSTEST0036
[TestClass]
public class ImageProcessingPerformanceTests : TestBase
{
    private ImageProcessingService _service = null!;
    private StorageService _storage = null!;
    private string _testPrefix = null!;
    private static readonly SKColor[] TestColors = [SKColors.Red, SKColors.Green, SKColors.Blue, SKColors.Gold, SKColors.Fuchsia];

    [TestInitialize]
    public new async Task CreateServer()
    {
        await base.CreateServer();
        _service = GetService<ImageProcessingService>();
        _storage = GetService<StorageService>();
        _testPrefix = $"perf-test-{Guid.NewGuid():N}";
    }

    [TestCleanup]
    public new async Task CleanServer()
    {
        await base.CleanServer();
    }

    [TestMethod]
    public async Task CompressPng_MaintainsFormat()
    {
        var path = await CreateTestImageAsync("test.png", SKEncodedImageFormat.Png, 800, 600);
        var result = await _service.CompressAsync(path, 400, 0);
        AssertIsValidImage(result);
        Assert.AreEqual(".png", Path.GetExtension(result).ToLowerInvariant());
        using var bmp = SKBitmap.Decode(result);
        Assert.AreEqual(400, bmp.Width);
        Assert.AreEqual(300, bmp.Height);
    }

    [TestMethod]
    public async Task Compress_WidthOnly_PreservesAspectRatio()
    {
        var path = await CreateTestImageAsync("wide.png", SKEncodedImageFormat.Png, 800, 400);
        var result = await _service.CompressAsync(path, 400, 0);
        using var bmp = SKBitmap.Decode(result);
        Assert.AreEqual(400, bmp.Width);
        Assert.AreEqual(200, bmp.Height);
    }

    [TestMethod]
    public async Task EdgeCase_CorruptFile_ReturnsOriginal()
    {
        var corruptData = new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x00, 0x00, 0x00 };
        var logicalPath = await CreateRawFileAsync("corrupt.png", corruptData);
        var physicalPath = _storage.GetFilePhysicalPath(logicalPath);

        var compressResult = await _service.CompressAsync(logicalPath, 100, 0);
        var clearResult = await _service.ClearExifAsync(logicalPath);

        Assert.AreEqual(physicalPath, compressResult);
        Assert.AreEqual(physicalPath, clearResult);
    }

    [TestMethod]
    public async Task Concurrency_SameExactTarget_SerializesCorrectly()
    {
        var path = await CreateTestImageAsync("serial.png", SKEncodedImageFormat.Png, 400, 300);
        var tasks = Enumerable.Range(0, 5).Select(_ => _service.CompressAsync(path, 200, 0)).ToArray();
        var results = await Task.WhenAll(tasks);

        var firstResult = results[0];
        foreach (var result in results)
        {
            Assert.AreEqual(firstResult, result);
            AssertIsValidImage(result);
        }
    }

    [TestMethod]
    public async Task Performance_TinyImage()
    {
        var path = await CreateTestImageAsync("tiny.png", SKEncodedImageFormat.Png, 100, 100);
        var sw = Stopwatch.StartNew();
        var result = await _service.CompressAsync(path, 50, 0);
        sw.Stop();

        AssertIsValidImage(result);
        Assert.IsTrue(sw.ElapsedMilliseconds < 2000,
            $"Tiny image compression took {sw.ElapsedMilliseconds}ms, expected < 2000ms");
    }

    private async Task<string> CreateTestImageAsync(string fileName, SKEncodedImageFormat format, int width, int height)
    {
        var logicalPath = $"{_testPrefix}/{fileName}";
        var physicalPath = _storage.GetFilePhysicalPath(logicalPath);
        var dir = Path.GetDirectoryName(physicalPath);
        if (!Directory.Exists(dir)) Directory.CreateDirectory(dir!);

        using var bitmap = new SKBitmap(width, height);
        using var canvas = new SKCanvas(bitmap);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var r = (byte)((x * 255) / width);
                var g = (byte)((y * 255) / height);
                var b = (byte)(((x + y) * 127) / (width + height));
                bitmap.SetPixel(x, y, new SKColor(r, g, b));
            }
        }

        var paint = new SKPaint();
        paint.IsAntialias = false;
        using (paint)
        {
            var rectW = Math.Max(1, width / 5);
            var rectH = Math.Max(1, height / 5);
            for (int i = 0; i < 5; i++)
            {
                paint.Color = TestColors[i];
                canvas.DrawRect(i * rectW, i * rectH, rectW, rectH, paint);
            }
        }
        canvas.Flush();

        using var image = SKImage.FromBitmap(bitmap);
        var encodeFormat = IsEncodable(format) ? format : SKEncodedImageFormat.Png;
        using var data = image.Encode(encodeFormat, 90)
            ?? throw new InvalidOperationException($"SkiaSharp Encode returned null for {encodeFormat}");
        await using var fs = File.Create(physicalPath);
        data.SaveTo(fs);

        return logicalPath;
    }

    private static bool IsEncodable(SKEncodedImageFormat format) => format switch
    {
        SKEncodedImageFormat.Png => true,
        SKEncodedImageFormat.Jpeg => true,
        SKEncodedImageFormat.Webp => true,
        _ => false
    };

    private async Task<string> CreateRawFileAsync(string fileName, byte[] data)
    {
        var logicalPath = $"{_testPrefix}/{fileName}";
        var physicalPath = _storage.GetFilePhysicalPath(logicalPath);
        var dir = Path.GetDirectoryName(physicalPath);
        if (!Directory.Exists(dir)) Directory.CreateDirectory(dir!);
        await File.WriteAllBytesAsync(physicalPath, data);
        return logicalPath;
    }

    private static void AssertIsValidImage(string path, string? message = null)
    {
        Assert.IsTrue(File.Exists(path), message ?? $"File should exist: {path}");
        using var codec = SKCodec.Create(path);
        Assert.IsNotNull(codec, message ?? $"File should be a valid image: {path}");
    }
}
#pragma warning restore MSTEST0036
