all:
	gcc chatbots.c -o chatbots $(pkg-config --cflags --libs gtk4 libadwaita-1 libsecret-1) -I/usr/local/runtime/include/ -L/usr/local/runtime/lib -lopenvino_genai_c

clean:
	rm -f chatbots
