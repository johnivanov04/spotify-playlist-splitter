const { defineConfig } = require("vitest/config");

module.exports = defineConfig({
  test: {
    environment: "node",
    globals: true,
    include: ["tests/**/*.test.mjs"],
    setupFiles: ["./tests/setup.mjs"],
    server: {
      deps: {
        // Inline app source so Vitest's loader processes it and vi.mock
        // can intercept require() calls from CJS modules.
        inline: [/^(?!.*node_modules).*/],
      },
    },
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "json-summary"],
      include: ["index.js", "lib/**/*.js", "db/schema.js"],
      exclude: [
        "tests/**",
        "db/migrations/**",
        "node_modules/**",
        "vitest.config.js",
        "drizzle.config.js",
      ],
    },
  },
});
