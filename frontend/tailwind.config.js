/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        'medical-blue': '#1E3A5F',
        'medical-light': '#E8F4FC',
        'medical-accent': '#0077B6',
        'medical-dark': '#0A1628',
        'medical-teal': '#00A896',
      }
    },
  },
  plugins: [],
}

