import jax
import jax.core
import jax._src.core
import jax.tree_util
import jax.util

# 1. Hot patch missing classes moved from jax.core to jax._src.core
missing_classes = [
    'Jaxpr', 'JaxprEqn', 'Literal', 'Var', 'DropVar', 
    'ClosedJaxpr', 'ShapedArray', 'Value', 'MainTrace', 'Trace',
    'Primitive'
]
for cls_name in missing_classes:
    if hasattr(jax._src.core, cls_name):
        setattr(jax.core, cls_name, getattr(jax._src.core, cls_name))

# 2. Hot patch tree_multimap (deprecated and removed in newer JAX)
if not hasattr(jax.tree_util, 'tree_multimap'):
    jax.tree_util.tree_multimap = jax.tree_util.tree_map

# 3. Hot patch safe_map and safe_zip for jax.util
def custom_safe_map(f, *args):
    return list(map(f, *args))

def custom_safe_zip(*args):
    return list(zip(*args))

if not hasattr(jax.util, 'safe_map'):
    jax.util.safe_map = custom_safe_map
if not hasattr(jax.util, 'safe_zip'):
    jax.util.safe_zip = custom_safe_zip
