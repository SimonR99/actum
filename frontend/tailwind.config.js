/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Hanken Grotesk", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"]
      },
      colors: {
        // Graphite — cool, slightly desaturated neutral ink ramp.
        slate: {
          50: "#F7F8FA",
          100: "#EEF0F3",
          200: "#E3E6EB",
          300: "#CDD2DA",
          400: "#98A0AB",
          500: "#6A727E",
          600: "#4C535E",
          700: "#353B44",
          800: "#22262D",
          900: "#15181D",
          950: "#0B0D11"
        },
        // Cobalt — the single sharp accent, reserved for active / primary.
        blue: {
          50: "#EEF1FE",
          100: "#DCE3FD",
          200: "#BCC9FB",
          500: "#3157FF",
          600: "#1F44F5",
          700: "#1736D2"
        },
        // Refined semantic tones.
        emerald: {
          50: "#E6F4EF",
          200: "#BBE3D5",
          400: "#2FB791",
          500: "#11A37B",
          600: "#0E8C6B",
          700: "#0B6F55"
        },
        red: {
          50: "#FBEBE9",
          200: "#F3C9C4",
          600: "#CE3A2F"
        },
        rose: {
          600: "#D8443A",
          700: "#BC3329"
        },
        amber: {
          50: "#FBF1E1",
          200: "#F0DBAE",
          700: "#9A6313"
        }
      },
      boxShadow: {
        panel: "0 1px 2px rgba(16, 19, 28, 0.04), 0 18px 36px -20px rgba(16, 19, 28, 0.20)",
        lift: "0 1px 2px rgba(16, 19, 28, 0.05), 0 22px 44px -22px rgba(16, 19, 28, 0.28)"
      },
      letterSpacing: {
        label: "0.14em"
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        },
        "live-ping": {
          "0%": { boxShadow: "0 0 0 0 rgba(17, 163, 123, 0.45)" },
          "70%": { boxShadow: "0 0 0 6px rgba(17, 163, 123, 0)" },
          "100%": { boxShadow: "0 0 0 0 rgba(17, 163, 123, 0)" }
        }
      },
      animation: {
        rise: "rise 0.55s cubic-bezier(0.16, 1, 0.3, 1) both",
        "live-ping": "live-ping 2.4s ease-out infinite"
      }
    }
  },
  plugins: []
};
