#include <gtk/gtk.h>
#include <adwaita.h>
#include <signal.h>
#include <string.h>
#include <libsecret/secret.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <time.h>
#include <openvino/genai/c/llm_pipeline.h>

static GtkBox *chat_box;
static GtkEntry *entry;
static GtkWidget *current_bubble;
static AdwOverlaySplitView *split_view;
static GtkWidget *main_stack;
static GtkListBox *chat_list;
static GtkWidget *send_button;
static GtkScrolledWindow *chat_scrolled;
static GtkWidget *select_button;
static gboolean selection_mode = FALSE;
static char *current_chat_id = NULL;
static ov_genai_llm_pipeline* llm_pipe = NULL;
static GMutex response_mutex;
static GString *current_response = NULL;

static const SecretSchema chat_schema = {
    "com.example.Chatbot.Chat",
    SECRET_SCHEMA_NONE,
    {
        { "chat_id", SECRET_SCHEMA_ATTRIBUTE_STRING },
        { "NULL", 0 }
    }
};

static void ensure_chats_dir() {
    mkdir("./chats", 0700);
}

static char* get_current_date_id() {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char *date_str = g_strdup_printf("%04d-%02d-%02d_%02d-%02d-%02d",
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec);
    return date_str;
}

static void save_chat_to_keyring(const char *chat_id, const char *content) {
    secret_password_store_sync(&chat_schema, SECRET_COLLECTION_DEFAULT,
        chat_id, content, NULL, NULL,
        "chat_id", chat_id, NULL);
}

static char* load_chat_from_keyring(const char *chat_id) {
    return secret_password_lookup_sync(&chat_schema, NULL, NULL,
        "chat_id", chat_id, NULL);
}

static void autoscroll_to_bottom() {
    GtkAdjustment *adj = gtk_scrolled_window_get_vadjustment(chat_scrolled);
    gtk_adjustment_set_value(adj, gtk_adjustment_get_upper(adj) - gtk_adjustment_get_page_size(adj));
}

static void append_message_start(gboolean is_user) {
    GtkWidget *bubble = gtk_label_new("");
    gtk_label_set_wrap(GTK_LABEL(bubble), TRUE);
    gtk_label_set_xalign(GTK_LABEL(bubble), 0);
    gtk_label_set_selectable(GTK_LABEL(bubble), TRUE);
    GtkWidget *frame = gtk_frame_new(NULL);
    gtk_frame_set_child(GTK_FRAME(frame), bubble);
    if (is_user) gtk_widget_set_halign(frame, GTK_ALIGN_END);
    else gtk_widget_set_halign(frame, GTK_ALIGN_START);
    gtk_widget_add_css_class(bubble, "chat-bubble");
    gtk_box_append(chat_box, frame);
    current_bubble = bubble;
    
    g_idle_add((GSourceFunc)autoscroll_to_bottom, NULL);
}

static void append_message_update(const char *text) {
    gtk_label_set_text(GTK_LABEL(current_bubble), text);
    g_idle_add((GSourceFunc)autoscroll_to_bottom, NULL);
}

typedef struct {
    char *user_text;
    GtkWidget *bubble;
} ChatData;

typedef struct {
    GtkWidget *bubble;
    char *text;
} StreamUpdate;

static gboolean stream_update_ui(gpointer data) {
    StreamUpdate *update = (StreamUpdate*)data;
    gtk_label_set_text(GTK_LABEL(update->bubble), update->text);
    autoscroll_to_bottom();
    g_free(update->text);
    g_free(update);
    return FALSE;
}

static ov_status_e stream_callback(void* user_data, const char* word) {
    GtkWidget *bubble = (GtkWidget*)user_data;
    
    g_mutex_lock(&response_mutex);
    g_string_append(current_response, word);
    char *text = g_strdup(current_response->str);
    g_mutex_unlock(&response_mutex);
    
    StreamUpdate *update = g_new(StreamUpdate, 1);
    update->bubble = bubble;
    update->text = text;
    g_idle_add(stream_update_ui, update);
    
    return ov_status_e_OK;
}

static gboolean enable_send_button(gpointer data) {
    gtk_widget_set_sensitive(GTK_WIDGET(data), TRUE);
    return FALSE;
}

static void save_current_chat() {
    if (!current_chat_id) return;
    
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "./chats/%s", current_chat_id);
    
    FILE *f = fopen(filepath, "w");
    if (!f) return;
    
    GtkWidget *child = gtk_widget_get_first_child(GTK_WIDGET(chat_box));
    while (child != NULL) {
        if (GTK_IS_FRAME(child)) {
            GtkWidget *bubble = gtk_frame_get_child(GTK_FRAME(child));
            if (GTK_IS_LABEL(bubble)) {
                const char *text = gtk_label_get_text(GTK_LABEL(bubble));
                gboolean is_user = (gtk_widget_get_halign(child) == GTK_ALIGN_END);
                fprintf(f, "%s: %s\n", is_user ? "User" : "Assistant", text);
            }
        }
        child = gtk_widget_get_next_sibling(child);
    }
    
    fclose(f);
}

static gpointer generate_thread_func(gpointer data) {
    ChatData *chat_data = (ChatData*)data;
    
    if (llm_pipe) {
        g_mutex_lock(&response_mutex);
        if (current_response) {
            g_string_free(current_response, TRUE);
        }
        current_response = g_string_new("");
        g_mutex_unlock(&response_mutex);
        
        ov_genai_generation_config* config = NULL;
        ov_genai_generation_config_create(&config);
        ov_genai_generation_config_set_max_new_tokens(config, 256);
        
        streamer_callback streamer;
        streamer.callback = stream_callback;
        streamer.user_data = chat_data->bubble;
        
        ov_genai_decoded_results* results = NULL;
        ov_genai_llm_pipeline_generate(llm_pipe, chat_data->user_text, config, &streamer, &results);
        
        if (results) {
            ov_genai_decoded_results_free(results);
        }
        
        ov_genai_generation_config_free(config);
    }
    
    g_free(chat_data->user_text);
    g_free(chat_data);
    
    g_idle_add((GSourceFunc)save_current_chat, NULL);
    g_idle_add(enable_send_button, send_button);
    
    return NULL;
}

static void send_message(GtkWidget *widget, gpointer user_data) {
    const char *text = gtk_editable_get_text(GTK_EDITABLE(entry));
    if (!text || strlen(text) == 0) return;
    
    gtk_widget_set_sensitive(send_button, FALSE);
    
    char *text_copy = g_strdup(text);
    
    append_message_start(TRUE);
    append_message_update(text_copy);
    gtk_editable_set_text(GTK_EDITABLE(entry), "");
    
    append_message_start(FALSE);
    append_message_update("");
    
    ChatData *chat_data = g_new(ChatData, 1);
    chat_data->user_text = text_copy;
    chat_data->bubble = current_bubble;
    
    GThread *thread = g_thread_new("generate-worker", generate_thread_func, chat_data);
    g_thread_unref(thread);
}

static gpointer init_genai(gpointer data) {
    ov_genai_llm_pipeline_create("gemma-2b-int4", "GPU", &llm_pipe);
    if (llm_pipe) {
        ov_genai_llm_pipeline_start_chat(llm_pipe);
    }
    return NULL;
}

static void toggle_selection_mode(GtkButton *button, gpointer user_data);

static void new_chat_clicked(GtkButton *button, gpointer user_data) {
    save_current_chat();
    
    GtkWidget *child = gtk_widget_get_first_child(GTK_WIDGET(chat_box));
    while (child != NULL) {
        GtkWidget *next = gtk_widget_get_next_sibling(child);
        gtk_box_remove(chat_box, child);
        child = next;
    }
    
    if (current_chat_id) g_free(current_chat_id);
    current_chat_id = get_current_date_id();
    
    GtkWidget *row = gtk_list_box_row_new();
    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
    gtk_widget_set_margin_start(box, 12);
    gtk_widget_set_margin_end(box, 12);
    gtk_widget_set_margin_top(box, 8);
    gtk_widget_set_margin_bottom(box, 8);
    
    GtkWidget *icon = gtk_image_new_from_icon_name("user-available-symbolic");
    gtk_box_append(GTK_BOX(box), icon);
    
    GtkWidget *label = gtk_label_new(current_chat_id);
    gtk_widget_set_hexpand(label, TRUE);
    gtk_label_set_xalign(GTK_LABEL(label), 0);
    gtk_box_append(GTK_BOX(box), label);
    
    gtk_list_box_row_set_child(GTK_LIST_BOX_ROW(row), box);
    gtk_list_box_insert(chat_list, row, 0);
    
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "./chats/%s", current_chat_id);
    FILE *f = fopen(filepath, "w");
    if (f) fclose(f);
    
    if (llm_pipe) {
        ov_genai_llm_pipeline_finish_chat(llm_pipe);
        ov_genai_llm_pipeline_start_chat(llm_pipe);
    }
}

static void load_saved_chats() {
    ensure_chats_dir();
    
    DIR *dir = opendir("./chats");
    if (!dir) return;
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        
        GtkWidget *row = gtk_list_box_row_new();
        GtkWidget *box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
        gtk_widget_set_margin_start(box, 12);
        gtk_widget_set_margin_end(box, 12);
        gtk_widget_set_margin_top(box, 8);
        gtk_widget_set_margin_bottom(box, 8);
        
        GtkWidget *icon = gtk_image_new_from_icon_name("user-available-symbolic");
        gtk_box_append(GTK_BOX(box), icon);
        
        GtkWidget *label = gtk_label_new(entry->d_name);
        gtk_widget_set_hexpand(label, TRUE);
        gtk_label_set_xalign(GTK_LABEL(label), 0);
        gtk_box_append(GTK_BOX(box), label);
        
        gtk_list_box_row_set_child(GTK_LIST_BOX_ROW(row), box);
        gtk_list_box_append(chat_list, row);
    }
    closedir(dir);
}

static void show_settings(GtkButton *button, gpointer user_data) {
    gtk_stack_set_visible_child_name(GTK_STACK(main_stack), "settings");
}

static void show_chat(GtkButton *button, gpointer user_data) {
    gtk_stack_set_visible_child_name(GTK_STACK(main_stack), "chat");
}

static void toggle_sidebar(GtkButton *button, gpointer user_data) {
    AdwOverlaySplitView *split = ADW_OVERLAY_SPLIT_VIEW(user_data);
    gboolean showing = adw_overlay_split_view_get_show_sidebar(split);
    adw_overlay_split_view_set_show_sidebar(split, !showing);
}

static void delete_selected_chats(GtkButton *button, gpointer user_data) {
    GtkWidget *row = gtk_widget_get_first_child(GTK_WIDGET(chat_list));
    
    while (row != NULL) {
        GtkWidget *next = gtk_widget_get_next_sibling(row);
        
        if (GTK_IS_LIST_BOX_ROW(row)) {
            GtkWidget *child = gtk_list_box_row_get_child(GTK_LIST_BOX_ROW(row));
            if (GTK_IS_BOX(child)) {
                GtkWidget *checkbox = gtk_widget_get_first_child(child);
                if (GTK_IS_CHECK_BUTTON(checkbox) && 
                    gtk_check_button_get_active(GTK_CHECK_BUTTON(checkbox))) {
                    
                    GtkWidget *label_widget = gtk_widget_get_next_sibling(
                        gtk_widget_get_next_sibling(checkbox));
                    if (GTK_IS_LABEL(label_widget)) {
                        const char *chat_name = gtk_label_get_text(GTK_LABEL(label_widget));
                        
                        char filepath[512];
                        snprintf(filepath, sizeof(filepath), "./chats/%s", chat_name);
                        remove(filepath);
                    }
                    
                    gtk_list_box_remove(chat_list, row);
                }
            }
        }
        
        row = next;
    }
    
    selection_mode = FALSE;
    gtk_button_set_label(GTK_BUTTON(select_button), "Select");
    
    row = gtk_widget_get_first_child(GTK_WIDGET(chat_list));
    while (row != NULL) {
        GtkWidget *next = gtk_widget_get_next_sibling(row);
        
        if (GTK_IS_LIST_BOX_ROW(row)) {
            GtkWidget *child = gtk_list_box_row_get_child(GTK_LIST_BOX_ROW(row));
            if (GTK_IS_BOX(child)) {
                GtkWidget *checkbox = gtk_widget_get_first_child(child);
                if (GTK_IS_CHECK_BUTTON(checkbox)) {
                    gtk_box_remove(GTK_BOX(child), checkbox);
                }
            }
        }
        
        row = next;
    }
    
    g_signal_handlers_disconnect_by_func(select_button, delete_selected_chats, user_data);
    g_signal_connect(select_button, "clicked", G_CALLBACK(toggle_selection_mode), user_data);
}

static void toggle_selection_mode(GtkButton *button, gpointer user_data) {
    selection_mode = !selection_mode;
    
    if (selection_mode) {
        gtk_button_set_icon_name(GTK_BUTTON(button), "user-trash-symbolic");
        gtk_button_set_label(GTK_BUTTON(button), "");
        
        g_signal_handlers_disconnect_by_func(button, toggle_selection_mode, user_data);
        g_signal_connect(button, "clicked", G_CALLBACK(delete_selected_chats), user_data);
        
        GtkWidget *row = gtk_widget_get_first_child(GTK_WIDGET(chat_list));
        while (row != NULL) {
            if (GTK_IS_LIST_BOX_ROW(row)) {
                GtkWidget *child = gtk_list_box_row_get_child(GTK_LIST_BOX_ROW(row));
                if (GTK_IS_BOX(child)) {
                    GtkWidget *checkbox = gtk_check_button_new();
                    gtk_box_prepend(GTK_BOX(child), checkbox);
                }
            }
            row = gtk_widget_get_next_sibling(row);
        }
    } else {
        gtk_button_set_icon_name(GTK_BUTTON(button), "");
        gtk_button_set_label(GTK_BUTTON(button), "Select");
        
        g_signal_handlers_disconnect_by_func(button, delete_selected_chats, user_data);
        g_signal_connect(button, "clicked", G_CALLBACK(toggle_selection_mode), user_data);
        
        GtkWidget *row = gtk_widget_get_first_child(GTK_WIDGET(chat_list));
        while (row != NULL) {
            if (GTK_IS_LIST_BOX_ROW(row)) {
                GtkWidget *child = gtk_list_box_row_get_child(GTK_LIST_BOX_ROW(row));
                if (GTK_IS_BOX(child)) {
                    GtkWidget *checkbox = gtk_widget_get_first_child(child);
                    if (GTK_IS_CHECK_BUTTON(checkbox)) {
                        gtk_box_remove(GTK_BOX(child), checkbox);
                    }
                }
            }
            row = gtk_widget_get_next_sibling(row);
        }
    }
}

static void on_activate(GtkApplication *app, gpointer user_data) {
    ensure_chats_dir();
    
    GtkWidget *win = adw_application_window_new(app);
    gtk_window_set_default_size(GTK_WINDOW(win), 800, 600);
    gtk_window_set_title(GTK_WINDOW(win), "Chatbot");
    
    split_view = ADW_OVERLAY_SPLIT_VIEW(adw_overlay_split_view_new());
    adw_overlay_split_view_set_sidebar_position(split_view, GTK_PACK_START);
    adw_overlay_split_view_set_max_sidebar_width(split_view, 300);
    adw_overlay_split_view_set_min_sidebar_width(split_view, 200);
    
    GtkWidget *sidebar_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    
    GtkWidget *sidebar_header = adw_header_bar_new();
    adw_header_bar_set_show_end_title_buttons(ADW_HEADER_BAR(sidebar_header), FALSE);
    select_button = gtk_button_new_with_label("Select");
    g_signal_connect(select_button, "clicked", G_CALLBACK(toggle_selection_mode), NULL);
    adw_header_bar_pack_start(ADW_HEADER_BAR(sidebar_header), select_button);
    gtk_box_append(GTK_BOX(sidebar_box), sidebar_header);
    
    GtkWidget *scrolled_sidebar = gtk_scrolled_window_new();
    gtk_widget_set_vexpand(scrolled_sidebar, TRUE);
    chat_list = GTK_LIST_BOX(gtk_list_box_new());
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scrolled_sidebar), GTK_WIDGET(chat_list));
    gtk_box_append(GTK_BOX(sidebar_box), scrolled_sidebar);
    
    GtkWidget *settings_button = gtk_button_new();
    GtkWidget *settings_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
    gtk_widget_set_margin_start(settings_box, 12);
    gtk_widget_set_margin_end(settings_box, 12);
    gtk_widget_set_margin_top(settings_box, 8);
    gtk_widget_set_margin_bottom(settings_box, 8);
    GtkWidget *settings_icon = gtk_image_new_from_icon_name("emblem-system-symbolic");
    GtkWidget *settings_label = gtk_label_new("Settings");
    gtk_widget_set_hexpand(settings_label, TRUE);
    gtk_label_set_xalign(GTK_LABEL(settings_label), 0);
    gtk_box_append(GTK_BOX(settings_box), settings_icon);
    gtk_box_append(GTK_BOX(settings_box), settings_label);
    gtk_button_set_child(GTK_BUTTON(settings_button), settings_box);
    gtk_box_append(GTK_BOX(sidebar_box), settings_button);
    
    adw_overlay_split_view_set_sidebar(split_view, sidebar_box);
    
    main_stack = gtk_stack_new();
    
    GtkWidget *chat_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    
    GtkWidget *header = adw_header_bar_new();
    adw_header_bar_set_title_widget(ADW_HEADER_BAR(header), gtk_label_new("Chatbot"));
    
    GtkWidget *menu_button = gtk_button_new_from_icon_name("open-menu-symbolic");
    g_signal_connect(menu_button, "clicked", G_CALLBACK(toggle_sidebar), split_view);
    adw_header_bar_pack_start(ADW_HEADER_BAR(header), menu_button);
    
    GtkWidget *new_chat_button = gtk_button_new_from_icon_name("list-add-symbolic");
    g_signal_connect(new_chat_button, "clicked", G_CALLBACK(new_chat_clicked), NULL);
    adw_header_bar_pack_end(ADW_HEADER_BAR(header), new_chat_button);
    
    gtk_box_append(GTK_BOX(chat_page), header);
    
    GtkWidget *scrolled = gtk_scrolled_window_new();
    chat_scrolled = GTK_SCROLLED_WINDOW(scrolled);
    gtk_widget_set_vexpand(scrolled, TRUE);
    gtk_widget_set_margin_top(scrolled, 12);
    gtk_widget_set_margin_bottom(scrolled, 12);
    gtk_widget_set_margin_start(scrolled, 12);
    gtk_widget_set_margin_end(scrolled, 12);
    gtk_box_append(GTK_BOX(chat_page), scrolled);
    
    GtkWidget *chat = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    chat_box = GTK_BOX(chat);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scrolled), chat);
    
    GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gtk_widget_set_hexpand(hbox, TRUE);
    gtk_widget_set_margin_top(hbox, 12);
    gtk_widget_set_margin_bottom(hbox, 12);
    gtk_widget_set_margin_start(hbox, 12);
    gtk_widget_set_margin_end(hbox, 12);
    gtk_box_append(GTK_BOX(chat_page), hbox);
    
    entry = GTK_ENTRY(gtk_entry_new());
    gtk_widget_set_hexpand(GTK_WIDGET(entry), TRUE);
    gtk_box_append(GTK_BOX(hbox), GTK_WIDGET(entry));
    
    send_button = gtk_button_new_from_icon_name("document-send-symbolic");
    gtk_box_append(GTK_BOX(hbox), send_button);
    g_signal_connect(send_button, "clicked", G_CALLBACK(send_message), NULL);
    g_signal_connect(entry, "activate", G_CALLBACK(send_message), NULL);
    
    gtk_stack_add_named(GTK_STACK(main_stack), chat_page, "chat");
    
    GtkWidget *settings_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    
    GtkWidget *settings_header = adw_header_bar_new();
    adw_header_bar_set_title_widget(ADW_HEADER_BAR(settings_header), gtk_label_new("Settings"));
    GtkWidget *back_button = gtk_button_new_from_icon_name("go-previous-symbolic");
    g_signal_connect(back_button, "clicked", G_CALLBACK(show_chat), NULL);
    adw_header_bar_pack_start(ADW_HEADER_BAR(settings_header), back_button);
    gtk_box_append(GTK_BOX(settings_page), settings_header);
    
    GtkWidget *settings_content = gtk_label_new("Settings content goes here");
    gtk_widget_set_vexpand(settings_content, TRUE);
    gtk_box_append(GTK_BOX(settings_page), settings_content);
    
    gtk_stack_add_named(GTK_STACK(main_stack), settings_page, "settings");
    
    g_signal_connect(settings_button, "clicked", G_CALLBACK(show_settings), NULL);
    
    adw_overlay_split_view_set_content(split_view, main_stack);
    adw_application_window_set_content(ADW_APPLICATION_WINDOW(win), GTK_WIDGET(split_view));
    
    gtk_window_present(GTK_WINDOW(win));
    
    GtkCssProvider *provider = gtk_css_provider_new();
    gtk_css_provider_load_from_string(provider,
        ".chat-bubble {"
        "background-color: #e0e0e0;"
        "border-radius: 12px;"
        "padding: 8px;"
        "max-width: 70%;"
        "border: none;"
        "}"
        "frame {"
        "border: none;"
        "}"
        "headerbar {"
        "background: @window_bg_color;"
        "box-shadow: none;"
        "}"
        "list {"
        "background: transparent;"
        "}"
        "button {"
        "padding: 6px;"
        "}"
    );
    gtk_style_context_add_provider_for_display(gdk_display_get_default(),
        GTK_STYLE_PROVIDER(provider),
        GTK_STYLE_PROVIDER_PRIORITY_USER);
    
    load_saved_chats();
    
    if (!current_chat_id) {
        new_chat_clicked(NULL, NULL);
    }
    
    g_mutex_init(&response_mutex);
    g_thread_new("genai-init", init_genai, NULL);
}

int main(int argc, char *argv[]) {
    signal(SIGINT, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    AdwApplication *app = adw_application_new("com.example.Chatbot", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(on_activate), NULL);
    int status = g_application_run(G_APPLICATION(app), argc, argv);
    
    if (llm_pipe) {
        ov_genai_llm_pipeline_finish_chat(llm_pipe);
        ov_genai_llm_pipeline_free(llm_pipe);
    }
    
    if (current_response) {
        g_string_free(current_response, TRUE);
    }
    
    g_mutex_clear(&response_mutex);
    
    return status;
}
