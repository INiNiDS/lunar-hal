/** @type {import('tailwindcss').Config} */
module.exports = {
  // Dioxus RSX macros live inside .rs source strings, so scan those directly.
  // Avoid dynamically concatenated class names in rsx! -- Tailwind can't see them here.
  content: ["./src/**/*.rs", "./index.html"],
  theme: {
    extend: {
      // Palette lifted from assets/main.css CSS custom properties, so both
      // stylesheets agree on the same "Rich Black + Liquid Glass" tokens.
      colors: {
        "bg-0": "#010204",
        "bg-1": "#0c0e14",
        "bg-2": "#11141c",
        "bg-3": "#181c28",
        "bg-4": "#1f2535",
        accent: {
          DEFAULT: "#6ea8ff",
          dim: "rgba(110, 168, 255, 0.15)",
          strong: "#4d8aff",
        },
        pinn: { DEFAULT: "#ff7a59", dim: "rgba(255, 122, 89, 0.12)" },
        gnn: { DEFAULT: "#b486ff", dim: "rgba(180, 134, 255, 0.12)" },
        siren: { DEFAULT: "#5cd3a8", dim: "rgba(92, 211, 168, 0.12)" },
        ok: "#5cd3a8",
        warn: "#f0b350",
        err: "#ff6b6b",
      },
      fontFamily: {
        display: ["Cinzel", "serif"],
        grotesk: ["Space Grotesk", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      keyframes: {
        "lamp-ignite": {
          "0%": { opacity: 0 },
          "12%": { opacity: 0.9 },
          "18%": { opacity: 0.1 },
          "28%": { opacity: 1 },
          "34%": { opacity: 0.35 },
          "100%": { opacity: 1 },
        },
        "desktop-reveal": {
          "0%": { opacity: 0, transform: "translateY(18px) scale(0.98)", filter: "brightness(0.2)" },
          "100%": { opacity: 1, transform: "translateY(0) scale(1)", filter: "brightness(1)" },
        },
        "led-pulse": {
          "0%, 100%": { opacity: 1 },
          "50%": { opacity: 0.3 },
        },
        "halo-breathe": {
          "0%, 100%": { opacity: 0.75, transform: "translateX(-50%) scale(1)" },
          "50%": { opacity: 1, transform: "translateX(-50%) scale(1.08)" },
        },
        "dock-app-reveal": {
          "0%": { opacity: 0, transform: "translateY(14px) scale(0.8)" },
          "100%": { opacity: 1, transform: "translateY(0) scale(1)" },
        },
        "window-open": {
          "0%": { opacity: 0, transform: "scale(0.97) translateY(10px)" },
          "100%": { opacity: 1, transform: "scale(1) translateY(0)" },
        },
      },
      animation: {
        "lamp-ignite": "lamp-ignite 1.2s ease-out forwards",
        "desktop-reveal": "desktop-reveal 1.4s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "led-pulse": "led-pulse 1.4s ease-in-out infinite",
        "halo-breathe": "halo-breathe 4s ease-in-out infinite",
        "dock-app-reveal": "dock-app-reveal 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "window-open": "window-open 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards",
      },
    },
  },
  plugins: [
    function ({ addComponents }) {
      addComponents({
        // ── Liquid Glass ───────────────────────────────────────────────────
        // On near-black backgrounds a plain blur reads as "slightly grey box",
        // so every glass surface also carries an inset top highlight (the
        // "lit edge") and a real drop shadow to detach it from the room.
        ".glass": {
          backgroundColor: "rgba(14, 17, 24, 0.55)",
          backdropFilter: "blur(22px) saturate(150%)",
          WebkitBackdropFilter: "blur(22px) saturate(150%)",
          border: "1px solid rgba(255,255,255,0.10)",
          boxShadow:
            "inset 0 1px 0 rgba(255,255,255,0.10), 0 10px 30px rgba(0,0,0,0.45)",
        },
        ".glass-strong": {
          backgroundColor: "rgba(16, 19, 27, 0.72)",
          backdropFilter: "blur(32px) saturate(170%)",
          WebkitBackdropFilter: "blur(32px) saturate(170%)",
          border: "1px solid rgba(255,255,255,0.14)",
          boxShadow:
            "inset 0 1px 0 rgba(255,255,255,0.16), inset 0 -1px 0 rgba(0,0,0,0.45), 0 28px 70px rgba(0,0,0,0.65)",
        },

        // ── The room ───────────────────────────────────────────────────────
        ".room-floor": {
          position: "absolute",
          inset: "0",
          pointerEvents: "none",
          background:
            "radial-gradient(ellipse 58% 32% at 50% 86%, rgba(110,168,255,0.022), transparent 72%)",
        },
        ".room-vignette": {
          position: "absolute",
          inset: "0",
          pointerEvents: "none",
          background:
            "radial-gradient(ellipse at 50% 38%, transparent 26%, rgba(0,0,0,0.68) 72%, rgba(0,0,0,0.96) 100%)",
        },

        // ── Lamp ───────────────────────────────────────────────────────────
        // A real cone: a trapezoid clipped out of a soft vertical gradient and
        // screen-blended so it adds light instead of painting a grey shape.
        ".lamp-cone": {
          position: "absolute",
          top: "96px",
          left: "50%",
          transform: "translateX(-50%)",
          width: "min(720px, 78vw)",
          height: "min(520px, 56vh)",
          clipPath: "polygon(47.6% 0%, 52.4% 0%, 100% 100%, 0% 100%)",
          background:
            "linear-gradient(to bottom, rgba(225,238,255,0.20) 0%, rgba(168,201,255,0.065) 25%, rgba(124,163,232,0.018) 52%, rgba(90,130,200,0) 76%)",
          filter: "blur(22px)",
          mixBlendMode: "screen",
          pointerEvents: "none",
        },
        ".lamp-bulb": {
          position: "absolute",
          bottom: "-3px",
          left: "50%",
          width: "46px",
          height: "14px",
          borderRadius: "9999px",
          background:
            "radial-gradient(ellipse at center, rgba(255,255,255,0.95), rgba(190,215,255,0.55) 45%, rgba(150,190,255,0) 75%)",
          filter: "blur(3px)",
          pointerEvents: "none",
        },

        // ── LEDs ───────────────────────────────────────────────────────────
        // Two layers: a solid core plus a separate blurred halo, otherwise a
        // 10px dot never reads as an actual glowing lamp.
        ".led": {
          position: "relative",
          display: "inline-block",
          borderRadius: "9999px",
          backgroundColor: "currentColor",
          boxShadow: "inset 0 0 2px rgba(255,255,255,0.6)",
        },
        ".led::after": {
          content: '""',
          position: "absolute",
          inset: "-7px",
          borderRadius: "9999px",
          background:
            "radial-gradient(circle, currentColor 0%, transparent 68%)",
          opacity: "0.6",
          pointerEvents: "none",
        },

        // ── Scrollbars ─────────────────────────────────────────────────────
        ".scrollbar-thin": {
          scrollbarWidth: "thin",
          scrollbarColor: "rgba(255,255,255,0.16) transparent",
          "&::-webkit-scrollbar": { width: "8px", height: "8px" },
          "&::-webkit-scrollbar-track": { background: "transparent" },
          "&::-webkit-scrollbar-thumb": {
            background: "rgba(255,255,255,0.14)",
            borderRadius: "9999px",
          },
          "&::-webkit-scrollbar-thumb:hover": {
            background: "rgba(255,255,255,0.24)",
          },
        },
      });
    },
  ],
};
