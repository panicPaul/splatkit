@fieldwise_init
struct Float2(ImplicitlyCopyable):
    var x: Float32
    var y: Float32


@fieldwise_init
struct Float3(ImplicitlyCopyable):
    var x: Float32
    var y: Float32
    var z: Float32


@fieldwise_init
struct Float4(ImplicitlyCopyable):
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32


@always_inline
def div_round_up(n: Int, d: Int) -> Int:
    return (n + d - 1) // d
