import ipywidgets as widgets
from IPython.display import display
import threading
import time
import re
from datetime import datetime
from kubernetes import client, config

class KServeDeployer:
    """A ready-to-use Jupyter widget for deploying Serverless KServe InferenceServices with vLLM.
    
    This widget automatically detects checkpoint folders from PyTorchJob logs and deploys them
    as Serverless KServe InferenceServices optimized for Large Language Model inference using vLLM runtime.
    It uses OpenShift Knative Serving with Istio service mesh for advanced traffic management and scaling.
    
    Features:
    - Automatic checkpoint folder detection from PyTorchJob logs
    - PVC auto-detection from PyTorchJob specifications  
    - vLLM-optimized Serverless InferenceService specs with GPU resources
    - OpenShift Knative integration with passthrough traffic routing
    - Istio service mesh with sidecar injection and HTTP prober rewriting

    - Support for both running and completed PyTorchJobs
    
    Args:
        path_mapping (dict): Optional mapping of training paths to deployment paths, e.g.
            {'/opt/model-dir': 'pvc://shared-storage', '/training': 'pvc://models'}
    
    Examples:
        # Basic usage - detects checkpoint folders from PyTorchJob logs
        KServeDeployer()
        
        # With path mapping for different storage URIs
        KServeDeployer(path_mapping={'/mnt/shared': 'pvc://shared-storage'})
    """
    
    def __init__(self, path_mapping=None):
        self.path_mapping = path_mapping or {}  # Map training paths to deployment paths
        self.current_namespace = self._detect_current_namespace()
        self.log_monitor_thread = None
        self.stop_monitoring = False
        self.last_checkpoint_time = {}
        self.detected_checkpoints = set()  # Checkpoints found from logs
        self.service_watch_thread = None
        self.stop_service_watching = False
        self._build_ui()
        self.update_checkpoints_dropdown()
        self._update_namespace_dropdown()
        self._update_pytorchjob_dropdown()
        self._update_inferenceservice_dropdown(preserve_selection=False)
        
    def _detect_current_namespace(self):
        """Detects the current namespace from environment variables."""
        import os
        
        # Try to read the namespace from the service account token
        try:
            namespace_file = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
            if os.path.exists(namespace_file):
                with open(namespace_file, 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        
        # Fallback to environment variables or default
        return os.environ.get('POD_NAMESPACE', 'default')
        
    def _build_ui(self):
        """Constructs all the UI components of the widget."""
        self.kube_api_server = widgets.Text(
            value='https://kubernetes.default.svc',
            placeholder='Enter Kubernetes API Server URL',
            description='K8s API Server:',
            style={'description_width': 'initial'}
        )
        self.kube_token = widgets.Password(
            description='K8s Token:',
            placeholder='Enter Kubernetes Bearer Token',
            style={'description_width': 'initial'}
        )
        self.namespace_dropdown = widgets.Dropdown(
            options=['default'],
            value='default',
            description='Namespace:',
            style={'description_width': 'initial'}
        )
        self.pytorchjob_dropdown = widgets.Dropdown(
            options=[],
            description='PyTorchJob:',
            style={'description_width': 'initial'}
        )
        self.refresh_jobs_button = widgets.Button(
            description='Refresh Jobs',
            button_style='info',
            icon='refresh',
            layout=widgets.Layout(width='120px')
        )
        self.scan_checkpoints_button = widgets.Button(
            description='Scan for Checkpoint Folders',
            button_style='warning',
            icon='search',
            layout=widgets.Layout(width='180px')
        )
        self.monitor_logs_checkbox = widgets.Checkbox(
            value=False,
            description='Monitor PyTorchJob logs for new checkpoint folders',
            style={'description_width': 'initial'},
            layout={'width': 'max-content'}
        )
        self.log_status = widgets.HTML(
            value='<i>Log monitoring: Inactive</i>',
            style={'description_width': 'initial'}
        )
        self.checkpoints_dropdown = widgets.Dropdown(
            options=[],
            description='Checkpoint Folders:',
            style={'description_width': 'initial'}
        )
        self.inference_service_name = widgets.Text(
            value='my-inference-service',
            placeholder='Auto-generated from checkpoint folder',
            description='Service Name:',
            style={'description_width': 'initial'}
        )

        
        # InferenceService management section
        self.inferenceservice_dropdown = widgets.Dropdown(
            options=[],
            description='InferenceServices:',
            style={'description_width': 'initial'}
        )
        self.refresh_services_button = widgets.Button(
            description='Refresh Services',
            button_style='info',
            icon='refresh',
            layout=widgets.Layout(width='140px')
        )
        self.delete_service_button = widgets.Button(
            description='Delete Service',
            button_style='danger',
            icon='trash',
            layout=widgets.Layout(width='130px')
        )
        self.service_status_info = widgets.HTML(
            value='<i>Select an InferenceService to see its status</i>',
            style={'description_width': 'initial'}
        )
        self.watch_service_checkbox = widgets.Checkbox(
            value=False,
            description='Monitor selected InferenceService for status changes (5s polling)',
            style={'description_width': 'initial'},
            layout={'width': 'max-content'}
        )
        self.watch_status = widgets.HTML(
            value='<i>InferenceService monitoring: Inactive</i>',
            style={'description_width': 'initial'}
        )
        
        self.create_button = widgets.Button(
            description='Create InferenceService',
            button_style='success',
            icon='check'
        )
        self.output = widgets.Output()
        
        # Link buttons to their functions
        self.create_button.on_click(self._create_inference_service)
        self.refresh_jobs_button.on_click(self._refresh_jobs_button_click)
        self.scan_checkpoints_button.on_click(self._scan_checkpoints_button_click)
        self.refresh_services_button.on_click(self._refresh_services_button_click)
        self.delete_service_button.on_click(self._delete_service_button_click)
        
        # Link checkbox to enable/disable log monitoring
        self.monitor_logs_checkbox.observe(self._on_monitor_logs_change, names='value')
        
        # Link checkbox to enable/disable InferenceService monitoring
        self.watch_service_checkbox.observe(self._on_watch_service_change, names='value')
        
        # Link namespace dropdown to update PyTorchJob list
        self.namespace_dropdown.observe(self._on_namespace_change, names='value')
        
        # Link kubernetes credentials to refresh PyTorchJob list when filled
        self.kube_api_server.observe(self._on_credentials_change, names='value')
        self.kube_token.observe(self._on_credentials_change, names='value')
        
        # Link dropdown to update service name
        self.checkpoints_dropdown.observe(self._update_service_name_from_checkpoint, names='value')
        
        # Link InferenceService dropdown to update status
        self.inferenceservice_dropdown.observe(self._update_service_status, names='value')
        self.inferenceservice_dropdown.observe(self._on_inferenceservice_selection_change, names='value')
        
        # Create HBox for PyTorchJob dropdown and buttons
        self.pytorchjob_row = widgets.HBox([
            self.pytorchjob_dropdown,
            self.refresh_jobs_button,
            self.scan_checkpoints_button
        ])
        
        # Create HBox for InferenceService dropdown and buttons
        self.inferenceservice_row = widgets.HBox([
            self.inferenceservice_dropdown,
            self.refresh_services_button,
            self.delete_service_button
        ])
        
        # The VBox holds all our UI elements
        self.ui = widgets.VBox([
            self.kube_api_server, 
            self.kube_token, 
            self.namespace_dropdown,
            self.pytorchjob_row,
            self.monitor_logs_checkbox,
            self.log_status,
            self.checkpoints_dropdown, 
            self.inference_service_name,
            self.create_button,
            widgets.HTML('<hr style="margin: 20px 0;"><h3>📊 InferenceService Management</h3>'),
            self.inferenceservice_row,
            self.watch_service_checkbox,
            self.watch_status,
            self.service_status_info,
            self.output
        ])

    def _sanitize_kubernetes_name(self, name):
        """Sanitizes a name to comply with Kubernetes naming conventions.
        
        Kubernetes names must:
        - Contain only lowercase alphanumeric characters or hyphens
        - Start and end with an alphanumeric character
        - Be no more than 253 characters (we'll limit to 63 for service names)
        """
        import re
        
        if not name:
            return "inference-service"
        
        # Extract just the folder name from path
        folder_name = name.split('/')[-1] if '/' in name else name
        
        # Convert to lowercase
        sanitized = folder_name.lower()
        
        # Replace invalid characters with hyphens
        sanitized = re.sub(r'[^a-z0-9\-]', '-', sanitized)
        
        # Remove multiple consecutive hyphens
        sanitized = re.sub(r'-+', '-', sanitized)
        
        # Ensure it starts and ends with alphanumeric character
        sanitized = re.sub(r'^[^a-z0-9]+', '', sanitized)
        sanitized = re.sub(r'[^a-z0-9]+$', '', sanitized)
        
        # Ensure it's not empty after sanitization
        if not sanitized:
            sanitized = "inference-service"
        
        # Limit length to 63 characters (Kubernetes service name limit)
        if len(sanitized) > 63:
            sanitized = sanitized[:60] + "svc"  # Keep some indication it was truncated
        
        # Add prefix if it doesn't start with alphanumeric
        if not sanitized[0].isalnum():
            sanitized = "svc-" + sanitized[1:]
        
        return sanitized

    def _extract_checkpoint_directory(self, matched_path, pattern_index):
        """Extracts checkpoint directory from matched path, handling both files and directories."""
        import os
        
        # File patterns are indices 6-9 in our pattern list (0-based)
        # These patterns match individual files and we need to extract their directory
        file_pattern_indices = {6, 7, 8, 9}
        
        if pattern_index in file_pattern_indices:
            # For file patterns, extract the directory containing the file
            if matched_path.startswith('/'):
                # It's a full path to a file, get the directory
                directory = os.path.dirname(matched_path)
                if directory and directory != '/':
                    return directory
            else:
                # It's likely just a filename without path, skip it
                return None
        else:
            # For directory patterns, use the path as-is (it should already be a directory)
            # But ensure it doesn't end with a filename
            if matched_path and matched_path.startswith('/'):
                # Check if it looks like a directory path (no file extension at the end)
                if not re.search(r'\.[a-zA-Z]{2,4}$', matched_path):
                    return matched_path
            
        return None
    
    def _map_checkpoint_path(self, training_path):
        """Maps a training checkpoint path to deployment path."""
        # Apply user-defined path mapping
        for training_prefix, deployment_prefix in self.path_mapping.items():
            if training_path.startswith(training_prefix):
                return training_path.replace(training_prefix, deployment_prefix, 1)
        
        # Default: use original path (assume it's accessible to InferenceService)
        return training_path
    
    def find_checkpoints(self):
        """Returns checkpoints detected from PyTorchJob logs."""
        if not self.detected_checkpoints:
            return []
            
        # Map all detected checkpoints to deployment paths
        mapped_checkpoints = []
        for original_path in self.detected_checkpoints:
            mapped_path = self._map_checkpoint_path(original_path)
            mapped_checkpoints.append(mapped_path)
            
        return sorted(mapped_checkpoints)

    def update_checkpoints_dropdown(self):
        """Populates the dropdown with checkpoints detected from PyTorchJob logs."""
        checkpoints = self.find_checkpoints()
        
        if not checkpoints:
            self.checkpoints_dropdown.disabled = True
            self.create_button.disabled = True
            self.checkpoints_dropdown.options = []
            with self.output:
                print("No checkpoint folders detected yet.")
                print("💡 Start PyTorchJob log monitoring to automatically detect checkpoint folders.")
        else:
            self.checkpoints_dropdown.disabled = False
            self.create_button.disabled = False
            self.checkpoints_dropdown.options = checkpoints
            with self.output:
                print(f"✅ Found {len(checkpoints)} checkpoint folder(s) from PyTorchJob logs:")
                for cp in checkpoints:
                    print(f"  📁 {cp}")
                    
                if self.path_mapping:
                    print(f"\n📍 Applied path mapping: {self.path_mapping}")
                
                # Update service name
                self._update_service_name_from_checkpoint()
                
    def _update_namespace_dropdown(self):
        """Updates the namespace dropdown with common namespaces."""
        common_namespaces = ['default', 'kube-system', 'kubeflow', 'opendatahub', 'redhat-ods-applications']
        
        # Add current namespace if it's not in the common list
        if self.current_namespace and self.current_namespace not in common_namespaces:
            common_namespaces.insert(0, self.current_namespace)
            
        # Update the dropdown options first
        self.namespace_dropdown.options = common_namespaces
        
        # Set default value to current namespace (it's now guaranteed to be in the options)
        if self.current_namespace:
            self.namespace_dropdown.value = self.current_namespace
    
    def _get_selected_namespace(self):
        """Returns the currently selected namespace."""
        return self.namespace_dropdown.value
    
    def _update_pytorchjob_dropdown(self):
        """Updates the PyTorchJob dropdown with available jobs in the selected namespace."""
        try:
            # Get Kubernetes configuration
            api_server_url = self.kube_api_server.value.strip()
            api_token = self.kube_token.value.strip()
            
            if not api_server_url or not api_token:
                self.pytorchjob_dropdown.options = []
                if hasattr(self, 'output'):  # Check if output widget exists
                    with self.output:
                        print("Please provide Kubernetes API server URL and token to load PyTorchJobs")
                return
                
            # Configure Kubernetes client
            configuration = client.Configuration()
            configuration.host = api_server_url
            configuration.api_key['authorization'] = f"Bearer {api_token}"
            configuration.verify_ssl = False
            
            api_client = client.ApiClient(configuration)
            api = client.CustomObjectsApi(api_client)
            
            # Get PyTorchJobs from the selected namespace
            namespace = self._get_selected_namespace()
            pytorchjobs = api.list_namespaced_custom_object(
                group='kubeflow.org',
                version='v1',
                namespace=namespace,
                plural='pytorchjobs'
            )
            
            job_names = []
            for job in pytorchjobs.get('items', []):
                job_name = job['metadata']['name']
                status = job.get('status', {}).get('conditions', [])
                # Add status indicator
                if any(c.get('type') == 'Running' and c.get('status') == 'True' for c in status):
                    job_names.append(f"{job_name} (Running)")
                elif any(c.get('type') == 'Succeeded' and c.get('status') == 'True' for c in status):
                    job_names.append(f"{job_name} (Completed)")
                else:
                    job_names.append(f"{job_name} (Other)")
                    
            self.pytorchjob_dropdown.options = job_names
            
            # Provide user feedback
            if hasattr(self, 'output'):
                with self.output:
                    if job_names:
                        print(f"Found {len(job_names)} PyTorchJob(s) in namespace '{self._get_selected_namespace()}'")
                    else:
                        print(f"No PyTorchJobs found in namespace '{self._get_selected_namespace()}'")
                        print("Make sure PyTorchJobs exist and you have the correct permissions")
            
        except client.ApiException as e:
            self.pytorchjob_dropdown.options = []
            with self.output:
                if e.status == 401:
                    print("Authentication failed: Please check your Kubernetes token")
                elif e.status == 403:
                    print("Access denied: You don't have permission to list PyTorchJobs in this namespace")
                elif e.status == 404:
                    print("PyTorchJob resource not found: Training Operator may not be installed")
                else:
                    print(f"Kubernetes API error ({e.status}): {e.reason}")
        except Exception as e:
            self.pytorchjob_dropdown.options = []
            with self.output:
                if "connection" in str(e).lower():
                    print("Connection failed: Please check your Kubernetes API server URL")
                else:
                    print(f"Error fetching PyTorchJobs: {e}")
    
    def _on_namespace_change(self, change):
        """Handles namespace dropdown changes."""
        self._update_pytorchjob_dropdown()
        self._update_inferenceservice_dropdown(preserve_selection=False)
        
    def _on_credentials_change(self, change):
        """Handles kubernetes credentials changes."""
        # Only refresh if both credentials are provided
        if self.kube_api_server.value.strip() and self.kube_token.value.strip():
            self._update_pytorchjob_dropdown()
            self._update_inferenceservice_dropdown(preserve_selection=False)
    
    def _refresh_jobs_button_click(self, button):
        """Handles refresh jobs button click."""
        with self.output:
            print("Refreshing PyTorchJob list...")
        self._update_pytorchjob_dropdown()
    
    def _scan_checkpoints_button_click(self, button):
        """Handles scan checkpoints button click."""
        selected_job = self.pytorchjob_dropdown.value
        if not selected_job:
            with self.output:
                print("⚠️  Please select a PyTorchJob to scan for checkpoint folders")
            return
            
        # Extract job name (remove status indicator)
        job_name = selected_job.split(' (')[0]
        
        with self.output:
            print(f"🔍 Manually scanning PyTorchJob '{job_name}' for checkpoint folders...")
        
        self._scan_job_for_checkpoints(job_name)
    
    def _refresh_services_button_click(self, button):
        """Handles refresh services button click."""
        with self.output:
            print("Refreshing InferenceService list...")
        self._update_inferenceservice_dropdown()
    
    def _delete_service_button_click(self, button):
        """Handles delete service button click."""
        selected_service = self.inferenceservice_dropdown.value
        if not selected_service:
            with self.output:
                print("⚠️  Please select an InferenceService to delete")
            return
            
        service_name = selected_service.split(' (')[0]
        
        # Confirm deletion
        try:
            # Get Kubernetes configuration
            api_server_url = self.kube_api_server.value.strip()
            api_token = self.kube_token.value.strip()
            
            if not api_server_url or not api_token:
                with self.output:
                    print("❌ Kubernetes credentials required for deletion")
                return
                
            configuration = client.Configuration()
            configuration.host = api_server_url
            configuration.api_key['authorization'] = f"Bearer {api_token}"
            configuration.verify_ssl = False
            
            api_client = client.ApiClient(configuration)
            api = client.CustomObjectsApi(api_client)
            
            # Delete the InferenceService
            namespace = self._get_selected_namespace()
            with self.output:
                print(f"🗑️  Deleting InferenceService '{service_name}' in namespace '{namespace}'...")
            
            api.delete_namespaced_custom_object(
                group='serving.kserve.io',
                version='v1beta1',
                namespace=namespace,
                plural='inferenceservices',
                name=service_name
            )
            
            with self.output:
                print(f"✅ InferenceService '{service_name}' deleted successfully!")
                print("🔄 Refreshing service list...")
            
            # Refresh the list
            self._update_inferenceservice_dropdown()
            
        except client.ApiException as e:
            with self.output:
                if e.status == 404:
                    print(f"❌ InferenceService '{service_name}' not found (may have been already deleted)")
                elif e.status == 403:
                    print(f"❌ Access denied: You don't have permission to delete InferenceService '{service_name}'")
                else:
                    print(f"❌ Kubernetes API Error: {e.reason}")
        except Exception as e:
            with self.output:
                print(f"❌ Error deleting InferenceService: {e}")
    
    def _update_service_name_from_checkpoint(self, change=None):
        """Updates the InferenceService name based on the selected checkpoint folder."""
        try:
            checkpoint_path = self.checkpoints_dropdown.value
            
            if not checkpoint_path:
                # Reset to default if no checkpoint selected
                self.inference_service_name.value = 'my-inference-service'
                return
            
            # Generate sanitized service name from checkpoint path
            sanitized_name = self._sanitize_kubernetes_name(checkpoint_path)
            
            # Update the service name field
            self.inference_service_name.value = sanitized_name
            
            with self.output:
                print(f"📝 Auto-generated service name: '{sanitized_name}'")
                
        except Exception as e:
            with self.output:
                print(f"❌ Error generating service name: {e}")
    

    def _update_inferenceservice_dropdown(self, preserve_selection=True):
        """Updates the InferenceService dropdown with available services in the selected namespace."""
        try:
            # Store current selection to preserve it if requested
            current_selection = None
            current_service_name = None
            if preserve_selection and self.inferenceservice_dropdown.value:
                current_selection = self.inferenceservice_dropdown.value
                current_service_name = current_selection.split(' (')[0]
            
            # Get Kubernetes configuration
            api_server_url = self.kube_api_server.value.strip()
            api_token = self.kube_token.value.strip()
            
            if not api_server_url or not api_token:
                self.inferenceservice_dropdown.options = []
                return
                
            # Configure Kubernetes client
            configuration = client.Configuration()
            configuration.host = api_server_url
            configuration.api_key['authorization'] = f"Bearer {api_token}"
            configuration.verify_ssl = False
            
            api_client = client.ApiClient(configuration)
            api = client.CustomObjectsApi(api_client)
            
            # Get InferenceServices from the selected namespace
            namespace = self._get_selected_namespace()
            inference_services = api.list_namespaced_custom_object(
                group='serving.kserve.io',
                version='v1beta1',
                namespace=namespace,
                plural='inferenceservices'
            )
            
            service_names = []
            found_current_service = False
            new_selection = None
            
            for service in inference_services.get('items', []):
                service_name = service['metadata']['name']
                
                # Get status
                status = service.get('status', {})
                conditions = status.get('conditions', [])
                
                # Determine overall status
                ready_status = 'Unknown'
                for condition in conditions:
                    if condition.get('type') == 'Ready':
                        if condition.get('status') == 'True':
                            ready_status = 'Ready'
                        elif condition.get('status') == 'False':
                            ready_status = 'Not Ready'
                        break
                
                service_display_name = f"{service_name} ({ready_status})"
                service_names.append(service_display_name)
                
                # Check if this is the currently selected service
                if preserve_selection and current_service_name == service_name:
                    found_current_service = True
                    new_selection = service_display_name
                    
            self.inferenceservice_dropdown.options = service_names
            
            # Restore selection if the service still exists
            if preserve_selection and found_current_service and new_selection:
                self.inferenceservice_dropdown.value = new_selection
            elif preserve_selection and current_selection and not found_current_service:
                # Service was deleted - stop monitoring and clear selection
                if self.watch_service_checkbox.value:
                    with self.output:
                        print(f"🗑️  Monitored service '{current_service_name}' was deleted, stopping monitoring...")
                    self.watch_service_checkbox.value = False
                # Clear selection since service no longer exists
                if service_names:
                    self.inferenceservice_dropdown.value = service_names[0]
            
            # Update status for currently selected service
            if service_names and hasattr(self, 'service_status_info'):
                self._update_service_status()
            
        except client.ApiException as e:
            self.inferenceservice_dropdown.options = []
            if hasattr(self, 'output'):
                with self.output:
                    if e.status == 401:
                        print("Authentication failed: Please check your Kubernetes token")
                    elif e.status == 403:
                        print("Access denied: You don't have permission to list InferenceServices")
                    elif e.status == 404:
                        print("InferenceService resource not found: KServe may not be installed")
        except Exception as e:
            self.inferenceservice_dropdown.options = []
            if hasattr(self, 'output'):
                with self.output:
                    print(f"Error fetching InferenceServices: {e}")
    
    def _update_service_status(self, change=None):
        """Updates the service status display for the selected InferenceService."""
        try:
            selected_service = self.inferenceservice_dropdown.value
            if not selected_service:
                self.service_status_info.value = '<i>Select an InferenceService to see its status</i>'
                return
            
            service_name = selected_service.split(' (')[0]
            
            # Get Kubernetes configuration
            api_server_url = self.kube_api_server.value.strip()
            api_token = self.kube_token.value.strip()
            
            if not api_server_url or not api_token:
                self.service_status_info.value = '<i>Kubernetes credentials required for status</i>'
                return
                
            configuration = client.Configuration()
            configuration.host = api_server_url
            configuration.api_key['authorization'] = f"Bearer {api_token}"
            configuration.verify_ssl = False
            
            api_client = client.ApiClient(configuration)
            api = client.CustomObjectsApi(api_client)
            
            # Get specific InferenceService
            namespace = self._get_selected_namespace()
            service = api.get_namespaced_custom_object(
                group='serving.kserve.io',
                version='v1beta1',
                namespace=namespace,
                plural='inferenceservices',
                name=service_name
            )
            
            # Parse status information
            status = service.get('status', {})
            conditions = status.get('conditions', [])
            url = status.get('url', 'Not available')
            
            # Build status display
            status_html = f'''
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 6px; margin: 4px 0;">
                <strong>📊 Status for {service_name}:</strong><br/>
                <strong>🔗 URL:</strong> <code>{url}</code><br/>
                <strong>📋 Conditions:</strong><br/>
            '''
            
            for condition in conditions:
                condition_type = condition.get('type', 'Unknown')
                condition_status = condition.get('status', 'Unknown')
                reason = condition.get('reason', '')
                message = condition.get('message', '')
                
                if condition_status == 'True':
                    icon = '✅'
                    color = 'green'
                elif condition_status == 'False':
                    icon = '❌'
                    color = 'red'
                else:
                    icon = '⚠️'
                    color = 'orange'
                
                status_html += f'''
                <div style="margin-left: 15px; color: {color};">
                    {icon} <strong>{condition_type}:</strong> {condition_status}
                '''
                
                if reason:
                    status_html += f'<br/>&nbsp;&nbsp;&nbsp;&nbsp;<small><i>Reason: {reason}</i></small>'
                if message:
                    status_html += f'<br/>&nbsp;&nbsp;&nbsp;&nbsp;<small><i>{message}</i></small>'
                    
                status_html += '</div>'
            
            status_html += '</div>'
            self.service_status_info.value = status_html
            
        except client.ApiException as e:
            self.service_status_info.value = f'<i>Error getting service status: {e.reason}</i>'
        except Exception as e:
            self.service_status_info.value = f'<i>Error: {e}</i>'
        
    def _on_monitor_logs_change(self, change):
        """Handles log monitoring checkbox changes."""
        if change['new']:  # Start monitoring
            self._start_log_monitoring()
        else:  # Stop monitoring
            self._stop_log_monitoring()
    
    def _start_log_monitoring(self):
        """Starts background log monitoring for the selected PyTorchJob."""
        if self.log_monitor_thread and self.log_monitor_thread.is_alive():
            return  # Already monitoring
            
        selected_job = self.pytorchjob_dropdown.value
        if not selected_job:
            with self.output:
                print("Please select a PyTorchJob to monitor")
            self.monitor_logs_checkbox.value = False
            return
            
        # Extract job name (remove status indicator)
        job_name = selected_job.split(' (')[0]
        
        # First, do an immediate scan for completed jobs to find historical checkpoints
        with self.output:
            print(f"🔍 Starting initial checkpoint scan for PyTorchJob: {job_name}")
        self._scan_job_for_checkpoints(job_name)
        
        self.stop_monitoring = False
        self.log_monitor_thread = threading.Thread(
            target=self._monitor_logs_worker,
            args=(job_name,),
            daemon=True
        )
        self.log_monitor_thread.start()
        
        self.log_status.value = f'<span style="color: green;">Log monitoring: Active for {job_name}</span>'
        with self.output:
            print(f"Started continuous monitoring for PyTorchJob: {job_name}")
    
    def _stop_log_monitoring(self):
        """Stops background log monitoring."""
        self.stop_monitoring = True
        self.log_status.value = '<i>Log monitoring: Inactive</i>'
        with self.output:
            print("Stopped log monitoring")
    
    def _on_watch_service_change(self, change):
        """Handles InferenceService monitoring checkbox changes."""
        if change['new']:  # Start monitoring
            self._start_service_watching()
        else:  # Stop monitoring
            self._stop_service_watching()
    
    def _start_service_watching(self):
        """Starts background monitoring for the selected InferenceService."""
        if self.service_watch_thread and self.service_watch_thread.is_alive():
            return  # Already monitoring
            
        selected_service = self.inferenceservice_dropdown.value
        if not selected_service:
            with self.output:
                print("Please select an InferenceService to monitor")
            self.watch_service_checkbox.value = False
            return
            
        # Extract service name (remove status indicator)
        service_name = selected_service.split(' (')[0]
        
        with self.output:
            print(f"📊 Starting to monitor InferenceService: {service_name}")
        
        self.stop_service_watching = False
        self.service_watch_thread = threading.Thread(
            target=self._watch_service_worker,
            args=(service_name,),
            daemon=True
        )
        self.service_watch_thread.start()
        
        self.watch_status.value = f'<span style="color: green;">InferenceService monitoring: Active for {service_name}</span>'
    
    def _stop_service_watching(self):
        """Stops background InferenceService monitoring."""
        self.stop_service_watching = True
        self.watch_status.value = '<i>InferenceService monitoring: Inactive</i>'
        with self.output:
            print("Stopped InferenceService monitoring")
    
    def _on_inferenceservice_selection_change(self, change):
        """Handles InferenceService dropdown selection changes."""
        # If monitoring is currently active and a different service is selected, stop monitoring
        if self.watch_service_checkbox.value and self.service_watch_thread and self.service_watch_thread.is_alive():
            with self.output:
                print("🔄 InferenceService selection changed, stopping current monitoring...")
            self.watch_service_checkbox.value = False  # This will trigger _on_watch_service_change
    
    def _watch_service_worker(self, service_name):
        """Background worker that polls an InferenceService for status changes every 5 seconds."""
        try:
            # Get Kubernetes configuration
            api_server_url = self.kube_api_server.value
            api_token = self.kube_token.value
            namespace = self._get_selected_namespace()
            
            if not api_server_url or not api_token:
                with self.output:
                    print("❌ Cannot monitor service: Kubernetes credentials not provided")
                self.watch_service_checkbox.value = False
                return
            
            configuration = client.Configuration()
            configuration.host = api_server_url
            configuration.api_key['authorization'] = f"Bearer {api_token}"
            configuration.verify_ssl = False
            
            api_client = client.ApiClient(configuration)
            api = client.CustomObjectsApi(api_client)
            
            with self.output:
                print(f"📊 Starting to monitor InferenceService '{service_name}' in namespace '{namespace}'")
                print(f"🔄 Status refresh every 5 seconds")
            
            last_ready_status = None
            last_full_status = None  # Track full status object for better change detection
            periodic_refresh_interval = 5  # Refresh every 5 seconds
            verbose_periodic_refresh = False  # Set to True for debugging
            
            while not self.stop_service_watching:
                try:
                    # Get the current InferenceService directly
                    service = api.get_namespaced_custom_object(
                        group='serving.kserve.io',
                        version='v1beta1',
                        namespace=namespace,
                        plural='inferenceservices',
                        name=service_name
                    )
                    
                    # Extract status information
                    status = service.get('status', {})
                    conditions = status.get('conditions', [])
                    
                    # Find the Ready condition
                    ready_status = 'Unknown'
                    ready_reason = ''
                    ready_message = ''
                    
                    for condition in conditions:
                        if condition.get('type') == 'Ready':
                            ready_status = condition.get('status', 'Unknown')
                            ready_reason = condition.get('reason', '')
                            ready_message = condition.get('message', '')
                            break
                    
                    # Check for any status changes (not just Ready condition)
                    current_full_status = {
                        'ready_status': ready_status,
                        'ready_reason': ready_reason,
                        'ready_message': ready_message,
                        'url': status.get('url', ''),
                        'conditions_count': len(conditions)
                    }
                    
                    # Check if any status changed (more comprehensive than just ready_status)
                    status_changed = (current_full_status != last_full_status) or (ready_status != last_ready_status)
                    
                    if status_changed:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        
                        with self.output:
                            print(f"🔄 [{timestamp}] InferenceService '{service_name}' status changed:")
                            print(f"   Ready: {ready_status}")
                            if ready_reason:
                                print(f"   Reason: {ready_reason}")
                            if ready_message:
                                print(f"   Message: {ready_message}")
                        
                        # Always update the UI for any status change
                        with self.output:
                            print(f"🔄 Refreshing service information...")
                        
                        # Update the service status display
                        self._update_service_status()
                        
                        # Refresh the dropdown to show updated status
                        self._update_inferenceservice_dropdown()
                        
                        # Special celebration when service becomes ready
                        if ready_status == 'True' and last_ready_status != 'True':
                            with self.output:
                                print(f"🎉 [{timestamp}] InferenceService '{service_name}' is now READY!")
                            
                            # Get the service URL if available
                            service_url = status.get('url', '')
                            if service_url:
                                with self.output:
                                    print(f"🌐 Service URL: {service_url}")
                        
                        # Update last known status
                        last_ready_status = ready_status
                        last_full_status = current_full_status
                    elif verbose_periodic_refresh:
                        # Only show periodic refresh messages in debug mode
                        with self.output:
                            print(f"🔄 [{datetime.now().strftime('%H:%M:%S')}] Status check: {ready_status}")
                    
                    # Sleep for the refresh interval
                    time.sleep(periodic_refresh_interval)
                    
                except client.ApiException as e:
                    if e.status == 404:
                        with self.output:
                            print(f"❌ InferenceService '{service_name}' not found - service may have been deleted")
                        # Refresh the dropdown to reflect deletion
                        self._update_inferenceservice_dropdown()
                        self.stop_service_watching = True
                        break
                    else:
                        with self.output:
                            print(f"❌ API error monitoring '{service_name}' (status {e.status}): {e.reason}")
                        time.sleep(10)  # Wait longer on API errors
                        continue
                except Exception as e:
                    with self.output:
                        print(f"❌ Error monitoring '{service_name}': {e}")
                    time.sleep(10)  # Wait longer on general errors
                    continue
                    
        except Exception as e:
            with self.output:
                print(f"💥 Fatal error in service monitoring: {e}")
        finally:
            if not self.stop_service_watching:
                self.watch_status.value = '<span style="color: red;">InferenceService monitoring: Error</span>'
            with self.output:
                print(f"📊 Stopped monitoring InferenceService '{service_name}'")
    
    def _scan_job_for_checkpoints(self, job_name):
        """Immediately scans a PyTorchJob's complete log history for checkpoints."""
        try:
            # Get Kubernetes configuration
            api_server_url = self.kube_api_server.value
            api_token = self.kube_token.value
            namespace = self._get_selected_namespace()
            
            if not api_server_url or not api_token:
                with self.output:
                    print("⚠️  Cannot scan: Kubernetes credentials not provided")
                return
            
            configuration = client.Configuration()
            configuration.host = api_server_url
            configuration.api_key['authorization'] = f"Bearer {api_token}"
            configuration.verify_ssl = False
            
            api_client = client.ApiClient(configuration)
            v1 = client.CoreV1Api(api_client)
            
            # Checkpoint detection patterns - focus on directories only
            checkpoint_patterns = [
                # Directory patterns (common in transformers/pytorch lightning)
                r'saving.*?checkpoint.*?to\s+([/\w\-\./]+/checkpoint-\d+)',
                r'saved.*?checkpoint.*?to\s+([/\w\-\./]+/checkpoint-\d+)', 
                r'checkpoint.*?saved.*?to\s+([/\w\-\./]+)',
                r'saving.*?model.*?to\s+([/\w\-\./]+)',
                r'saving model checkpoint to\s+([/\w\-\./]+)',
                r'model.*?checkpoint.*?saved.*?to\s+([/\w\-\./]+)',
                # File patterns - but we'll extract the directory containing the file
                r'saved checkpoint.*?(\S+)\.(?:ckpt|pt|pth|bin|safetensors)',
                r'saving.*?checkpoint.*?(\S+)\.(?:ckpt|pt|pth|bin|safetensors)',
                r'checkpoint.*?saved.*?(\S+)\.(?:ckpt|pt|pth|bin|safetensors)',
                r'model.*?saved.*?(\S+)\.(?:ckpt|pt|pth|bin|safetensors)',
                # Generic patterns
                r'saving.*?(?:checkpoint|model).*?([/\w\-\./]+/(?:checkpoint|ckpt|step|epoch)[-_]\d+)',
                r'saved.*?(?:checkpoint|model).*?([/\w\-\./]+/(?:checkpoint|ckpt|step|epoch)[-_]\d+)'
            ]
            
            # Find pods associated with the PyTorchJob
            possible_selectors = [
                f"job-name={job_name}",
                f"pytorch-job-name={job_name}",
                f"training.kubeflow.org/job-name={job_name}",
                f"pytorch.org/job-name={job_name}",
                f"app.kubernetes.io/name={job_name}"
            ]
            
            pods_found = []
            for selector in possible_selectors:
                pods = v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=selector
                )
                if pods.items:
                    pods_found = pods.items
                    with self.output:
                        print(f"📦 Found {len(pods.items)} pod(s) for job '{job_name}' using selector: {selector}")
                    break
            
            if not pods_found:
                # Try name pattern matching
                all_pods = v1.list_namespaced_pod(namespace=namespace)
                for pod in all_pods.items:
                    if job_name in pod.metadata.name:
                        pods_found.append(pod)
                
                if pods_found:
                    with self.output:
                        print(f"📦 Found {len(pods_found)} pod(s) for job '{job_name}' by name pattern")
            
            if not pods_found:
                with self.output:
                    print(f"❌ No pods found for PyTorchJob '{job_name}'")
                return
            
            checkpoints_found_this_scan = set()
            
            for pod in pods_found:
                pod_name = pod.metadata.name
                
                try:
                    with self.output:
                        print(f"📋 Scanning complete log history of pod '{pod_name}'...")
                    
                    # Get all logs for this pod
                    logs = v1.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=namespace,
                        timestamps=True
                    )
                    
                    lines_scanned = 0
                    for line in logs.split('\n'):
                        lines_scanned += 1
                        line = line.strip()
                        if not line:
                            continue
                            
                        for i, pattern in enumerate(checkpoint_patterns):
                            match = re.search(pattern, line, re.IGNORECASE)
                            if match:
                                matched_path = match.group(1)
                                
                                # Convert to directory path if it's a file
                                checkpoint_path = self._extract_checkpoint_directory(matched_path, i)
                                
                                if checkpoint_path:  # Only proceed if we got a valid directory
                                    timestamp = datetime.now().strftime("%H:%M:%S")
                                    
                                    # Add to detected checkpoints if new
                                    if checkpoint_path not in self.detected_checkpoints:
                                        self.detected_checkpoints.add(checkpoint_path)
                                        checkpoints_found_this_scan.add(checkpoint_path)
                                        self.last_checkpoint_time[checkpoint_path] = timestamp
                                        
                                        with self.output:
                                            print(f"✅ Found checkpoint folder: {checkpoint_path}")
                                    else:
                                        checkpoints_found_this_scan.add(checkpoint_path)
                                break
                    
                    with self.output:
                        print(f"📊 Scanned {lines_scanned} log lines from pod '{pod_name}'")
                        
                except client.ApiException as e:
                    with self.output:
                        if e.status == 404:
                            print(f"⚠️  Pod '{pod_name}' logs not found (pod may be too old)")
                        else:
                            print(f"❌ Error accessing logs for pod '{pod_name}': {e}")
                except Exception as e:
                    with self.output:
                        print(f"❌ Error scanning pod '{pod_name}': {e}")
            
            # Update UI with results
            if checkpoints_found_this_scan:
                with self.output:
                    print(f"🎉 Initial scan complete: Found {len(checkpoints_found_this_scan)} checkpoint folder(s)")
                    for cp in sorted(checkpoints_found_this_scan):
                        print(f"  📁 {cp}")
                self.update_checkpoints_dropdown()
            else:
                with self.output:
                    print("📭 No checkpoint folders found in job logs")
                    
        except Exception as e:
            with self.output:
                print(f"💥 Error during checkpoint scan: {e}")
    
    def _monitor_logs_worker(self, job_name):
        """Background worker that monitors PyTorchJob logs for checkpoint events."""
        try:
            # Get Kubernetes configuration
            api_server_url = self.kube_api_server.value
            api_token = self.kube_token.value
            namespace = self._get_selected_namespace()
            
            configuration = client.Configuration()
            configuration.host = api_server_url
            configuration.api_key['authorization'] = f"Bearer {api_token}"
            configuration.verify_ssl = False
            
            api_client = client.ApiClient(configuration)
            v1 = client.CoreV1Api(api_client)
            
            with self.output:
                print(f"🔍 Starting log monitoring for PyTorchJob '{job_name}' in namespace '{namespace}'")
            
            # Checkpoint detection patterns - focus on directories only
            checkpoint_patterns = [
                # Directory patterns (common in transformers/pytorch lightning)
                r'saving.*?checkpoint.*?to\s+([/\w\-\./]+/checkpoint-\d+)',
                r'saved.*?checkpoint.*?to\s+([/\w\-\./]+/checkpoint-\d+)', 
                r'checkpoint.*?saved.*?to\s+([/\w\-\./]+)',
                r'saving.*?model.*?to\s+([/\w\-\./]+)',
                r'saving model checkpoint to\s+([/\w\-\./]+)',
                r'model.*?checkpoint.*?saved.*?to\s+([/\w\-\./]+)',
                # File patterns - but we'll extract the directory containing the file
                r'saved checkpoint.*?(\S+)\.(?:ckpt|pt|pth|bin|safetensors)',
                r'saving.*?checkpoint.*?(\S+)\.(?:ckpt|pt|pth|bin|safetensors)',
                r'checkpoint.*?saved.*?(\S+)\.(?:ckpt|pt|pth|bin|safetensors)',
                r'model.*?saved.*?(\S+)\.(?:ckpt|pt|pth|bin|safetensors)',
                # Generic patterns
                r'saving.*?(?:checkpoint|model).*?([/\w\-\./]+/(?:checkpoint|ckpt|step|epoch)[-_]\d+)',
                r'saved.*?(?:checkpoint|model).*?([/\w\-\./]+/(?:checkpoint|ckpt|step|epoch)[-_]\d+)'
            ]
            
            # Get pods associated with the PyTorchJob - Try multiple label selectors
            possible_selectors = [
                f"job-name={job_name}",
                f"pytorch-job-name={job_name}",
                f"training.kubeflow.org/job-name={job_name}",
                f"pytorch.org/job-name={job_name}",
                f"app.kubernetes.io/name={job_name}"
            ]
            
            logs_inspected = False
            pods_found = False
            
            while not self.stop_monitoring:
                try:
                    # Try different label selectors to find pods
                    all_pods = []
                    working_selector = None
                    
                    for selector in possible_selectors:
                        pods = v1.list_namespaced_pod(
                            namespace=namespace,
                            label_selector=selector
                        )
                        if pods.items:
                            all_pods = pods.items
                            working_selector = selector
                            if not pods_found:
                                with self.output:
                                    print(f"📦 Found {len(pods.items)} pod(s) using label selector: {selector}")
                                pods_found = True
                            break
                    
                    if not all_pods and not pods_found:
                        # Try to find pods without label selector, matching by name pattern
                        all_pods_in_ns = v1.list_namespaced_pod(namespace=namespace)
                        for pod in all_pods_in_ns.items:
                            if job_name in pod.metadata.name:
                                all_pods.append(pod)
                        
                        if all_pods:
                            with self.output:
                                print(f"📦 Found {len(all_pods)} pod(s) by name pattern matching")
                            pods_found = True
                    
                    if not all_pods:
                        if not pods_found:
                            with self.output:
                                print(f"⚠️  No pods found for PyTorchJob '{job_name}'. Tried selectors: {possible_selectors}")
                            pods_found = True  # Avoid repeated messages
                        time.sleep(10)
                        continue
                    
                    for pod in all_pods:
                        if self.stop_monitoring:
                            break
                            
                        pod_name = pod.metadata.name
                        
                        try:
                            # For the monitoring worker, only check recent logs since we do initial scan on start
                            # This prevents repeated scanning of completed jobs
                            logs = v1.read_namespaced_pod_log(
                                name=pod_name,
                                namespace=namespace,
                                since_seconds=60,  # Last 60 seconds
                                timestamps=True
                            )
                            
                            if not logs_inspected:
                                with self.output:
                                    print(f"📋 Inspecting logs from pod '{pod_name}'...")
                                    logs_inspected = True
                            
                            # Check for checkpoint patterns
                            lines_checked = 0
                            for line in logs.split('\n'):
                                if self.stop_monitoring:
                                    break
                                
                                lines_checked += 1
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                for i, pattern in enumerate(checkpoint_patterns):
                                    match = re.search(pattern, line, re.IGNORECASE)
                                    if match:
                                        matched_path = match.group(1)
                                        
                                        # Convert to directory path if it's a file
                                        checkpoint_path = self._extract_checkpoint_directory(matched_path, i)
                                        
                                        if checkpoint_path:  # Only proceed if we got a valid directory
                                            timestamp = datetime.now().strftime("%H:%M:%S")
                                            
                                            with self.output:
                                                print(f"✅ [Pattern {i+1}] Found checkpoint pattern in log line: {line}")
                                                print(f"📁 Extracted checkpoint folder: {checkpoint_path}")
                                            
                                            # Avoid duplicate notifications for the same checkpoint
                                            if checkpoint_path not in self.last_checkpoint_time:
                                                self.last_checkpoint_time[checkpoint_path] = timestamp
                                                
                                                # Add to detected checkpoints for prioritized listing
                                                self.detected_checkpoints.add(checkpoint_path)
                                                
                                                # Update UI
                                                with self.output:
                                                    print(f"🎉 [{timestamp}] New checkpoint folder detected: {checkpoint_path}")
                                                
                                                # Refresh checkpoints dropdown
                                                self.update_checkpoints_dropdown()
                                            else:
                                                with self.output:
                                                    print(f"🔄 [{timestamp}] Checkpoint folder already known: {checkpoint_path}")
                                        break
                            
                            # Log inspection summary every 30 seconds
                            current_time = datetime.now()
                            if not hasattr(self, '_last_summary_time'):
                                self._last_summary_time = current_time
                            elif (current_time - self._last_summary_time).seconds >= 30:
                                with self.output:
                                    print(f"📊 Log inspection summary: {lines_checked} lines checked, {len(self.detected_checkpoints)} total checkpoint folders found")
                                self._last_summary_time = current_time
                                        
                        except client.ApiException as e:
                            # Pod might not be ready yet or logs not available
                            if e.status == 400:
                                with self.output:
                                    print(f"⚠️  Pod '{pod_name}' logs not available yet (container may be starting)")
                            pass
                        except Exception as e:
                            with self.output:
                                print(f"❌ Error reading logs from pod '{pod_name}': {e}")
                            
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    with self.output:
                        print(f"❌ Error in log monitoring: {e}")
                    time.sleep(10)  # Wait longer on error
                    
        except Exception as e:
            with self.output:
                print(f"💥 Fatal error in log monitoring: {e}")
        finally:
            if not self.stop_monitoring:
                self.log_status.value = '<span style="color: red;">Log monitoring: Error</span>'

    def _create_inference_service(self, b):
        """Creates an InferenceService object when the button is clicked."""
        with self.output:
            self.output.clear_output()
            print('Creating InferenceService...')

        try:
            # Get values from widgets
            api_server_url = self.kube_api_server.value
            api_token = self.kube_token.value
            checkpoint_path = self.checkpoints_dropdown.value
            service_name = self.inference_service_name.value
            namespace = self._get_selected_namespace()

            if not all([api_server_url, api_token, checkpoint_path, service_name, namespace]):
                 with self.output:
                    self.output.clear_output()
                    print("Error: All fields must be filled out.")
                 return

            # --- Kubernetes API Configuration ---
            configuration = client.Configuration()
            configuration.host = api_server_url
            configuration.api_key['authorization'] = f"Bearer {api_token}"
            configuration.verify_ssl = False # WARNING: Insecure, not for production
            
            api_client = client.ApiClient(configuration)
            api = client.CustomObjectsApi(api_client)

            # Get PVC name from the selected PyTorchJob
            selected_job = self.pytorchjob_dropdown.value
            if not selected_job:
                with self.output:
                    self.output.clear_output()
                    print("Error: Please select a PyTorchJob to get PVC information.")
                return
                
            job_name = selected_job.split(' (')[0]
            pvc_name = self._get_pytorchjob_pvc_name(job_name, api_client)
            
            if not pvc_name:
                with self.output:
                    self.output.clear_output()
                    print("❌ Error: Could not determine PVC name from PyTorchJob.")
                    print("   InferenceService creation requires proper PVC configuration in PyTorchJob.")
                return
            
            # Extract relative path from checkpoint_path for the PVC using PyTorchJob spec
            relative_path = self._extract_relative_path_for_pvc(checkpoint_path, job_name, api_client)
            
            if not relative_path:
                with self.output:
                    self.output.clear_output()
                    print("❌ Error: Could not extract relative path from checkpoint using PyTorchJob specification.")
                    print("   This usually indicates:")
                    print("   • Checkpoint path doesn't match PyTorchJob PVC mount path")
                    print("   • PyTorchJob doesn't have proper PVC volume mounts configured")
                    print("   • Training job and inference deployment have mismatched storage configuration")
                return
                
            storage_uri = f'pvc://{pvc_name}/{relative_path}'
            
            with self.output:
                print(f"📦 Using PVC: {pvc_name}")
                print(f"📁 Checkpoint relative path: {relative_path}")
                print(f"🔗 Final storage URI: {storage_uri}")

            # --- InferenceService Definition ---
            inference_service = {
                'apiVersion': 'serving.kserve.io/v1beta1',
                'kind': 'InferenceService',
                'metadata': {
                    'name': service_name,
                    'annotations': {
                        'serving.knative.openshift.io/enablePassthrough': 'true',
                        'serving.kserve.io/deploymentMode': 'Serverless',
                        'serving.kserve.io/stop': 'false',
                        'sidecar.istio.io/inject': 'true',
                        'sidecar.istio.io/rewriteAppHTTPProbers': 'true'
                    }
                },
                'spec': {
                    'predictor': {
                        'automountServiceAccountToken': False,
                        'maxReplicas': 1,
                        'minReplicas': 1,
                        'model': {
                            'modelFormat': {
                                'name': 'vLLM'
                            },
                            'name': '',
                            'resources': {
                                'limits': {
                                    'cpu': '10',
                                    'memory': '20Gi',
                                    'nvidia.com/gpu': '1'
                                },
                                'requests': {
                                    'cpu': '6',
                                    'memory': '16Gi',
                                    'nvidia.com/gpu': '1'
                                }
                            },
                            'runtime': 'example',
                            'storageUri': storage_uri
                        }
                    }
                }
            }

            api.create_namespaced_custom_object(
                group='serving.kserve.io',
                version='v1beta1',
                namespace=namespace,
                plural='inferenceservices',
                body=inference_service,
            )

            with self.output:
                self.output.clear_output()
                print(f"✅ InferenceService '{service_name}' created successfully!")
                print(f"🎯 Model Format: vLLM")
                print(f"📦 Storage URI: {storage_uri}")
                print(f"💾 Resources: 6-10 CPU, 16-20Gi Memory, 1 GPU")
                print(f"🚀 Deployment Mode: Serverless (with OpenShift Knative)")
                print(f"🔗 Istio Sidecar: Enabled with HTTP prober rewriting")
                print(f"🌐 Passthrough: Enabled for direct traffic routing")
                print(f"🏠 Deployed to namespace: {namespace}")
                print(f"🔄 Refreshing InferenceService list...")
            
            # Refresh the InferenceService list to show the new service
            self._update_inferenceservice_dropdown()
            
            # Auto-select the newly created service if it exists
            for option in self.inferenceservice_dropdown.options:
                if option.startswith(f"{service_name} ("):
                    self.inferenceservice_dropdown.value = option
                    with self.output:
                        print(f"✅ Auto-selected newly created service: {service_name}")
                    break

        except client.ApiException as e:
            with self.output:
                self.output.clear_output()
                print(f"Kubernetes API Error: {e.reason}\nBody: {e.body}")
        except Exception as e:
            with self.output:
                self.output.clear_output()
                print(f"An unexpected error occurred: {e}")
    
    def _get_pytorchjob_pvc_name(self, job_name, api_client):
        """Extracts PVC name from PyTorchJob specification."""
        try:
            api = client.CustomObjectsApi(api_client)
            namespace = self._get_selected_namespace()
            
            # Get the PyTorchJob
            pytorchjob = api.get_namespaced_custom_object(
                group='kubeflow.org',
                version='v1',
                namespace=namespace,
                plural='pytorchjobs',
                name=job_name
            )
            
            # Look for PVC references in the job spec
            spec = pytorchjob.get('spec', {})
            
            # Check different worker types (master, worker)
            for worker_type in ['pytorchReplicaSpecs', 'replicaSpecs']:
                replica_specs = spec.get(worker_type, {})
                
                for replica_type, replica_spec in replica_specs.items():
                    template = replica_spec.get('template', {})
                    pod_spec = template.get('spec', {})
                    volumes = pod_spec.get('volumes', [])
                    
                    # Look for PVC volumes
                    for volume in volumes:
                        pvc_claim = volume.get('persistentVolumeClaim', {})
                        if pvc_claim:
                            pvc_name = pvc_claim.get('claimName')
                            if pvc_name:
                                with self.output:
                                    print(f"🔍 Found PVC '{pvc_name}' in PyTorchJob '{job_name}'")
                                return pvc_name
            
            with self.output:
                print(f"⚠️  No PVC found in PyTorchJob '{job_name}' specification")
            return None
            
        except client.ApiException as e:
            with self.output:
                print(f"❌ Error getting PyTorchJob '{job_name}': {e}")
            return None
        except Exception as e:
            with self.output:
                print(f"❌ Error extracting PVC from PyTorchJob: {e}")
            return None
    
    def _get_pvc_mount_path_from_job(self, job_name, api_client):
        """Extracts PVC mount path from PyTorchJob specification."""
        try:
            api = client.CustomObjectsApi(api_client)
            namespace = self._get_selected_namespace()
            
            # Get the PyTorchJob
            pytorchjob = api.get_namespaced_custom_object(
                group='kubeflow.org',
                version='v1',
                namespace=namespace,
                plural='pytorchjobs',
                name=job_name
            )
            
            # Look for volume mounts in the job spec
            spec = pytorchjob.get('spec', {})
            
            # Check different worker types (master, worker)
            for worker_type in ['pytorchReplicaSpecs', 'replicaSpecs']:
                replica_specs = spec.get(worker_type, {})
                
                for replica_type, replica_spec in replica_specs.items():
                    template = replica_spec.get('template', {})
                    pod_spec = template.get('spec', {})
                    
                    # Get volumes and their PVC names
                    volumes = pod_spec.get('volumes', [])
                    pvc_volume_mapping = {}
                    
                    for volume in volumes:
                        pvc_claim = volume.get('persistentVolumeClaim', {})
                        if pvc_claim:
                            volume_name = volume.get('name')
                            pvc_name = pvc_claim.get('claimName')
                            if volume_name and pvc_name:
                                pvc_volume_mapping[volume_name] = pvc_name
                    
                    # Get containers and their volume mounts
                    containers = pod_spec.get('containers', [])
                    for container in containers:
                        volume_mounts = container.get('volumeMounts', [])
                        
                        for volume_mount in volume_mounts:
                            volume_name = volume_mount.get('name')
                            mount_path = volume_mount.get('mountPath')
                            
                            if volume_name in pvc_volume_mapping and mount_path:
                                with self.output:
                                    print(f"🔍 Found PVC mount: {pvc_volume_mapping[volume_name]} → {mount_path}")
                                return mount_path
            
            with self.output:
                print(f"⚠️  No PVC mount path found in PyTorchJob '{job_name}' specification")
            return None
            
        except client.ApiException as e:
            with self.output:
                print(f"❌ Error getting PyTorchJob mount path '{job_name}': {e}")
            return None
        except Exception as e:
            with self.output:
                print(f"❌ Error extracting mount path from PyTorchJob: {e}")
            return None
    
    def _extract_relative_path_for_pvc(self, checkpoint_path, job_name=None, api_client=None):
        """Extracts the relative path within PVC from absolute checkpoint path using PyTorchJob spec."""
        
        # Require PyTorchJob info for accurate path extraction
        if not job_name or not api_client:
            with self.output:
                print(f"❌ Cannot extract relative path: PyTorchJob name or API client not available")
            return None
            
        # Get mount path from PyTorchJob spec
        pvc_mount_path = self._get_pvc_mount_path_from_job(job_name, api_client)
        
        if not pvc_mount_path:
            with self.output:
                print(f"❌ Cannot extract relative path: No PVC mount path found in PyTorchJob '{job_name}'")
            return None
            
        if not checkpoint_path.startswith(pvc_mount_path):
            with self.output:
                print(f"❌ Checkpoint path '{checkpoint_path}' does not start with PyTorchJob mount path '{pvc_mount_path}'")
                print(f"   This indicates a mismatch between training and deployment configurations")
            return None
            
        # Extract relative path using PyTorchJob mount path
        relative_path = checkpoint_path[len(pvc_mount_path):]
        # Remove leading slash if present
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]
            
        if not relative_path:
            with self.output:
                print(f"❌ Extracted relative path is empty - checkpoint path equals mount path")
            return None
            
        with self.output:
            print(f"✅ Extracted relative path: '{relative_path}' from mount '{pvc_mount_path}'")
            
        return relative_path
    
    def __del__(self):
        """Cleanup when widget is destroyed."""
        self._stop_log_monitoring()
        self._stop_service_watching()
    
    def _ipython_display_(self):
        """Allows the object to be displayed directly in a notebook."""
        display(self.ui)