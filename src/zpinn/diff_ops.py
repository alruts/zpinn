from jax import grad

# single derivative operators
dx_fn = lambda fn: grad(fn, argnums=0)
dy_fn = lambda fn: grad(fn, argnums=1)
dz_fn = lambda fn: grad(fn, argnums=2)

# double derivative operators
dxx_fn = lambda fn: grad(grad(fn, argnums=0), argnums=0)
dyy_fn = lambda fn: grad(grad(fn, argnums=1), argnums=1)
dzz_fn = lambda fn: grad(grad(fn, argnums=2), argnums=2)

# laplace operator
laplace_fn = lambda fn: dxx_fn(fn) + dyy_fn(fn) + dzz_fn(fn)
