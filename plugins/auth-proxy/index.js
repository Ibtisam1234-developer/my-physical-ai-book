// Simple API proxy plugin for Docusaurus
// Proxies /api/auth/* requests to the auth server and /api/personalize to backend

module.exports = function (context, options) {
  return {
    name: 'docusaurus-auth-proxy',
    configureWebpack() {
      return {
        devServer: {
          setupMiddlewares: (middlewares, devServer) => {
            const { app } = devServer;

            // Proxy auth requests to auth server
            app.use('/api/auth', async (req, res, next) => {
              const authServerUrl = process.env.AUTH_SERVER_URL || 'http://localhost:3001';

              // Collect request body
              let body = '';
              req.on('data', chunk => {
                body += chunk.toString();
              });

              req.on('end', async () => {
                try {
                  const targetUrl = `${authServerUrl}${req.url}`;

                  const proxyRes = await fetch(targetUrl, {
                    method: req.method,
                    headers: {
                      'Content-Type': 'application/json',
                      ...req.headers,
                      'content-length': body.length,
                    },
                    body: body ? body : undefined,
                  });

                  // Forward response
                  res.status(proxyRes.status);
                  for (const [key, value] of proxyRes.headers.entries()) {
                    res.setHeader(key, value);
                  }

                  const proxyBody = await proxyRes.text();
                  res.send(proxyBody);
                } catch (error) {
                  console.error('Auth proxy error:', error);
                  res.status(500).json({ error: 'Auth proxy error' });
                }
              });
            });

            // Proxy personalize requests to backend
            app.use('/api/personalize', async (req, res, next) => {
              const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';

              // Collect request body
              let body = '';
              req.on('data', chunk => {
                body += chunk.toString();
              });

              req.on('end', async () => {
                try {
                  const targetUrl = `${backendUrl}${req.url}`;

                  const proxyRes = await fetch(targetUrl, {
                    method: req.method,
                    headers: {
                      'Content-Type': 'application/json',
                      ...req.headers,
                      'content-length': body.length,
                    },
                    body: body ? body : undefined,
                  });

                  // Forward response
                  res.status(proxyRes.status);
                  for (const [key, value] of proxyRes.headers.entries()) {
                    res.setHeader(key, value);
                  }

                  const proxyBody = await proxyRes.text();
                  res.send(proxyBody);
                } catch (error) {
                  console.error('Personalize proxy error:', error);
                  res.status(500).json({ error: 'Personalize proxy error' });
                }
              });
            });


            return middlewares;
          },
        },
      };
    },
  };
};
