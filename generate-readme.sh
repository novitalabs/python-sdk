#!/bin/bash

cat > README.md <<'EOF'
# Novita AI Python SDK

This SDK is based on the official [API documentation](https://docs.novita.ai/).

**Join our discord server for help:**

[![](https://dcbadge.vercel.app/api/server/Mqx7nWYzDF)](https://discord.com/invite/Mqx7nWYzDF)

## Installation

```bash
pip install novita-client
```

## Examples

- [fine tune example](https://colab.research.google.com/drive/1j_ii9TN67nuauvc3PiauwZnC2lT62tGF?usp=sharing)
EOF


for FILE in $(ls examples/ | grep py | sort -V); do
    NAME=$(echo "$FILE" | sed 's/.py//')
    echo "- [$NAME](./examples/$FILE)" >> README.md
done

echo "## Code Examples" >> README.md

for FILE in $(ls examples/ | grep py | sort -V); do
    NAME=$(echo "$FILE" | sed 's/.py//')
    echo "" >> README.md
    echo "### $NAME" >> README.md
    echo "\`\`\`python" >> README.md
    cat examples/$FILE >> README.md
    echo "\`\`\`" >> README.md
done