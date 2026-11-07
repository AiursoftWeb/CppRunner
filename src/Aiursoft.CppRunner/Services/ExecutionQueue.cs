using System.Threading.RateLimiting;

namespace Aiursoft.CppRunner.Services;

public sealed class ExecutionQueue : IDisposable
{
    private readonly ConcurrencyLimiter _limiter = new(new ConcurrencyLimiterOptions
    {
        PermitLimit = 2,
        QueueLimit = int.MaxValue,
        QueueProcessingOrder = QueueProcessingOrder.OldestFirst
    });

    public ValueTask<RateLimitLease> AcquireAsync(CancellationToken cancellationToken) =>
        _limiter.AcquireAsync(1, cancellationToken);

    public void Dispose() => _limiter.Dispose();
}
