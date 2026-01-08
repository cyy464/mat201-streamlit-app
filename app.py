\
import numpy as np
import sympy as sp
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# MAT201 Calculus Assignment 2
# Interactive Multivariable Calculus App
# Topics: Visualization, Partial Derivatives, Directional Derivatives, Gradient, Differentials
# -----------------------------

st.set_page_config(page_title="MAT201 Multivariable Calculus App", layout="wide")

st.title("MAT201: Multivariable Calculus Interactive App")
st.caption("Topics: function visualization, partial derivatives, directional derivatives, gradient & steepest ascent, differentials (linear approximation).")

# Symbols
x, y = sp.symbols("x y", real=True)

def parse_function(expr_text: str):
    """Parse a user function safely with a limited set of allowed symbols."""
    allowed = {
        "x": x, "y": y,
        "exp": sp.exp, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sqrt": sp.sqrt, "log": sp.log, "pi": sp.pi
    }
    expr = sp.sympify(expr_text, locals=allowed)
    return sp.simplify(expr)

def unit_vector(vx: float, vy: float):
    v = np.array([vx, vy], dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return None
    return v / n

def make_grid(a, b, xspan=2.0, yspan=2.0, n=120):
    xs = np.linspace(a - xspan, a + xspan, n)
    ys = np.linspace(b - yspan, b + yspan, n)
    return np.meshgrid(xs, ys)

# Sidebar inputs
st.sidebar.header("Inputs")

default_f = "x**2 + 2*y**2"
expr_text = st.sidebar.text_input("Enter f(x, y):", value=default_f, help="Use Python-style math, e.g. x**2 + 2*y**2 or 80*exp(-(x**2+2*y**2))+20")

a = st.sidebar.number_input("Point a (x-coordinate)", value=1.0, step=0.1, format="%.4f")
b = st.sidebar.number_input("Point b (y-coordinate)", value=1.0, step=0.1, format="%.4f")

st.sidebar.subheader("Direction vector v")
vx = st.sidebar.number_input("v_x", value=3.0, step=0.5, format="%.4f")
vy = st.sidebar.number_input("v_y", value=4.0, step=0.5, format="%.4f")

st.sidebar.subheader("Plot range")
xspan = st.sidebar.slider("x-range half-width", 0.5, 5.0, 2.5, 0.1)
yspan = st.sidebar.slider("y-range half-width", 0.5, 5.0, 2.5, 0.1)

# Parse & compute
try:
    f_expr = parse_function(expr_text)
except Exception as e:
    st.error(f"Cannot parse f(x,y). Please check your expression.\n\nDetails: {e}")
    st.stop()

fx_expr = sp.diff(f_expr, x)
fy_expr = sp.diff(f_expr, y)

f = sp.lambdify((x, y), f_expr, "numpy")
fx = sp.lambdify((x, y), fx_expr, "numpy")
fy = sp.lambdify((x, y), fy_expr, "numpy")

f_val = float(f(a, b))
fx_val = float(fx(a, b))
fy_val = float(fy(a, b))
grad = np.array([fx_val, fy_val], dtype=float)
grad_norm = float(np.linalg.norm(grad))

u = unit_vector(vx, vy)
if u is None:
    st.warning("Direction vector v cannot be (0,0). Please change v.")
    st.stop()

D_u = float(np.dot(grad, u))
u_star = grad / grad_norm if grad_norm != 0 else None

tabs = st.tabs(["1) Visualization", "2) Partial Derivatives", "3) Directional Derivative", "4) Gradient & Differentials"])

# --- Tab 1: Visualization
with tabs[0]:
    st.subheader("Meaning & Visualization of f(x,y)")
    X, Y = make_grid(a, b, xspan=xspan, yspan=yspan, n=120)
    Z = f(X, Y)

    col1, col2 = st.columns(2)

    with col1:
        fig = plt.figure(figsize=(6, 4.3))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=0, antialiased=True, alpha=0.9)
        ax.scatter([a], [b], [f_val], s=40)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y)")
        ax.set_title("3D Surface z = f(x,y)")
        st.pyplot(fig, clear_figure=True)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.3))
        cs = ax2.contour(X, Y, Z, levels=12)
        ax2.clabel(cs, inline=True, fontsize=7)
        ax2.scatter([a], [b], s=40)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Contour (Level Curves)")
        st.pyplot(fig2, clear_figure=True)

    st.markdown(
        r"""
**Interpretation:**  
- The 3D surface shows the height \(z=f(x,y)\).  
- The contour plot shows level curves where \(f(x,y)\) is constant.  
Moving on the plane changes the function value depending on direction and steepness.
"""
    )

# --- Tab 2: Partial derivatives
with tabs[1]:
    st.subheader("Partial Derivatives as Rates of Change")
    st.latex(r"f_x(a,b) = \frac{\partial f}{\partial x}(a,b), \quad f_y(a,b) = \frac{\partial f}{\partial y}(a,b)")
    st.write("Symbolic derivatives:")
    st.latex(rf"f_x(x,y) = {sp.latex(fx_expr)}")
    st.latex(rf"f_y(x,y) = {sp.latex(fy_expr)}")

    st.write("Values at the selected point:")
    st.metric("f(a,b)", f"{f_val:.6g}")
    c1, c2 = st.columns(2)
    c1.metric("f_x(a,b)", f"{fx_val:.6g}")
    c2.metric("f_y(a,b)", f"{fy_val:.6g}")

    st.markdown(
        r"""
**Meaning:**  
- \(f_x(a,b)\) is the instantaneous rate of change of \(f\) when moving in the \(+x\) direction (holding \(y\) constant).  
- \(f_y(a,b)\) is the instantaneous rate of change of \(f\) when moving in the \(+y\) direction (holding \(x\) constant).
"""
    )

# --- Tab 3: Directional derivative
with tabs[2]:
    st.subheader("Directional Derivative")
    st.latex(r"D_{\mathbf{u}}f(a,b)=\nabla f(a,b)\cdot \mathbf{u}")
    st.write("Your direction vector v and the corresponding unit vector u:")
    st.write(f"v = ({vx:.4g}, {vy:.4g})")
    st.write(f"u = ({u[0]:.6g}, {u[1]:.6g})")

    st.write("Result at the selected point:")
    st.metric(r"$D_{\mathbf{u}}f(a,b)$", f"{D_u:.6g}")

    st.markdown(
        r"""
**Meaning:**  
\(D_{\mathbf{u}}f(a,b)\) estimates how fast the function changes when you move from \((a,b)\) in direction \(\mathbf{u}\).
"""
    )

# --- Tab 4: Gradient & differentials
with tabs[3]:
    st.subheader("Gradient, Steepest Ascent, and Differentials")
    st.latex(r"\nabla f(a,b)=(f_x(a,b), f_y(a,b))")
    st.write(f"∇f(a,b) = ({grad[0]:.6g}, {grad[1]:.6g})")
    st.write(f"||∇f(a,b)|| = {grad_norm:.6g}")

    if u_star is not None:
        st.write(f"Direction of steepest ascent (unit gradient): ({u_star[0]:.6g}, {u_star[1]:.6g})")
        st.write("Maximum rate of increase equals ||∇f(a,b)||.")

    st.markdown("**Differentials / Linear Approximation:**")
    st.latex(r"df \approx f_x(a,b)\,dx + f_y(a,b)\,dy")
    st.latex(rf"z \approx {f_val:.6g} + ({fx_val:.6g})(x-{a:.4g}) + ({fy_val:.6g})(y-{b:.4g})")

    st.markdown(
        r"""
**Meaning:**  
For small movements \((dx,dy)\) near \((a,b)\), the change in the function is approximately \(df\).  
This is useful for fast estimation without computing the exact value of \(f(a+dx,b+dy)\).
"""
    )

    # Show vectors on contour
    X, Y = make_grid(a, b, xspan=xspan, yspan=yspan, n=120)
    Z = f(X, Y)
    fig3, ax3 = plt.subplots(figsize=(6, 4.3))
    cs = ax3.contour(X, Y, Z, levels=12)
    ax3.clabel(cs, inline=True, fontsize=7)
    ax3.scatter([a], [b], s=40)

    scale = 0.6 * min(xspan, yspan)
    ax3.arrow(a, b, u[0]*scale, u[1]*scale, head_width=0.08*scale, length_includes_head=True)
    ax3.text(a + u[0]*scale, b + u[1]*scale, "u", fontsize=10)

    if grad_norm != 0:
        g_unit = grad / grad_norm
        ax3.arrow(a, b, g_unit[0]*scale, g_unit[1]*scale, head_width=0.08*scale, length_includes_head=True)
        ax3.text(a + g_unit[0]*scale, b + g_unit[1]*scale, "∇f", fontsize=10)

    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("Contour with Direction u and Gradient ∇f")
    st.pyplot(fig3, clear_figure=True)
